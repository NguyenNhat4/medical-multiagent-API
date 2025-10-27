from pocketflow import Node
from utils.llm import call_llm
from utils.auth import APIOverloadException
from utils.parsing import parse_yaml_with_schema
from config.timeout_config import timeout_config
from utils.llm import (
    PROMPT_OQA_CLASSIFY_EN,
    PROMPT_OQA_COMPOSE_VI_WITH_SOURCES,
    PROMPT_OQA_CHITCHAT,
)
from utils.helpers import (
    format_kb_qa_list,
    get_score_threshold,
    format_conversation_history
)
from utils.role_enum import (
    PERSONA_BY_ROLE
)
from utils.knowledge_base.kb_oqa import (
    retrieve_oqa,
    retrieve_random_oqa,
    get_references_by_ids,
    format_references_numbered,
)
import logging

# Configure logging for this module with Vietnam timezone
from utils.timezone_utils import setup_vietnam_logging
from config.logging_config import logging_config

if logging_config.USE_VIETNAM_TIMEZONE:
    logger = setup_vietnam_logging(__name__, 
                                 level=getattr(logging, logging_config.LOG_LEVEL.upper()),
                                 format_str=logging_config.LOG_FORMAT)
else:
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, logging_config.LOG_LEVEL.upper()))

# ========== OQA Orthodontist Nodes ==========

class OQAIngestDefaults(Node):
    """Set default role `orthodontist` and normalize input."""
    def prep(self, shared):
        logger.info("🔍 [OQAIngest] PREP - Reading role and input from shared")
        role = shared.get("role") or "orthodontist"
        query = shared.get("input") or shared.get("query", "")
        conversation_history = shared.get("conversation_history", [])
        logger.info(f"🔍 [OQAIngest] PREP - Role: {role}, Query: '{str(query)[:80]}...', History: {len(conversation_history)} messages")
        return role, str(query).strip(), conversation_history

    def exec(self, inputs):
        logger.info("🔍 [OQAIngest] EXEC - Processing role and query for OQA")
        role, query, conversation_history = inputs
        result = {"role": role, "query": query, "conversation_history": conversation_history}
        logger.info(f"🔍 [OQAIngest] EXEC - Processed: role={role}, query_len={len(query)}")
        return result

    def post(self, shared, prep_res, exec_res):
        logger.info("🔍 [OQAIngest] POST - Saving role and query to shared")
        shared["role"] = exec_res["role"]
        shared["query"] = exec_res["query"]
        shared["conversation_history"] = exec_res["conversation_history"]
        logger.info(f"🔍 [OQAIngest] POST - Saved role: {exec_res['role']}, query: '{exec_res['query'][:50]}...'")
        return "default"


class OQAClassifyEN(Node):
    """Classify English input and produce English rag_questions for OQA."""
    def prep(self, shared):
        logger.info("🧠 [OQAClassify] PREP - Building classification prompt")
        query = shared.get("query", "").strip()
        role = shared.get("role", "orthodontist")
        conversation_history = shared.get("conversation_history", [])
        # format last 3 pairs
        lines = []
        for msg in conversation_history[-6:]:
            try:
                who = msg.get("role")
                content = msg.get("content", "")
                lines.append(f"- {who}: {content}")
            except Exception:
                continue
        formatted_history = "\n".join(lines)
        prompt = PROMPT_OQA_CLASSIFY_EN.format(query=query, role=role, conversation_history=formatted_history)
        logger.info(f"🧠 [OQAClassify] PREP - Query: '{query[:60]}...', Role: {role}, History: {len(lines)} context lines")
        return prompt

    def exec(self, prompt):
        logger.info("🧠 [OQAClassify] EXEC - Calling LLM for EN classification")
        # Log the exact prompt being sent to LLM
        try:
            logger.info("🧠 [OQAClassify] PROMPT (len=%d):\n%s", len(prompt) if isinstance(prompt, str) else 0, prompt)
        except Exception:
            pass
        try:
            resp = call_llm(prompt, fast_mode=True, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)
            logger.info(f"🧠 [OQAClassify] EXEC - Raw classification response length: {len(resp)} chars")
            logger.info(f"🧠 [OQAClassify] EXEC - Full API response:\n{resp}")
            
            result = parse_yaml_with_schema(
                resp,
                required_fields=["type"],
                optional_fields=["confidence", "reason", "rag_questions"],
                field_types={"type": str, "confidence": str, "reason": str, "rag_questions": list},
            )
            if result:
                logger.info(f"🧠 [OQAClassify] EXEC - Classification: {result.get('type')}, confidence: {result.get('confidence')}, rag_questions: {len(result.get('rag_questions', []))}")
                return result
            else:
                logger.warning(f"🧠 [OQAClassify] EXEC - YAML parsing failed for classification")
                
        except APIOverloadException:
            logger.warning("🧠 [OQAClassify] EXEC - API overloaded, falling back to chitchat")
            return {"type": "chitchat", "rag_questions": []}
        except Exception as e:
            logger.warning(f"🧠 [OQAClassify] EXEC - Classification failed: {e}, defaulting to medical_question")
        return {"type": "medical_question", "rag_questions": []}

    def post(self, shared, prep_res, exec_res):
        logger.info("🧠 [OQAClassify] POST - Saving classification results")
        shared["input_type"] = exec_res.get("type", "medical_question")
        shared["rag_questions"] = exec_res.get("rag_questions", [])
        t = shared["input_type"]
        logger.info(f"🧠 [OQAClassify] POST - Type: {t}, RAG questions: {len(shared['rag_questions'])}, routing to: {'chitchat' if t == 'chitchat' else 'retrieve_kb'}")
        if t == "chitchat":
            return "chitchat"
        return "retrieve_kb"


class OQARetrieve(Node):
    """Retrieve only from OQA vector index using English queries (user + rag)."""
    def prep(self, shared):
        logger.info("📚 [OQARetrieve] PREP - Reading query and rag_questions for OQA retrieval")
        query = shared.get("query", "")
        rag_questions = shared.get("rag_questions", [])
        logger.info(f"📚 [OQARetrieve] PREP - User query: '{query[:60]}...', RAG questions: {len(rag_questions) if rag_questions else 0}")
        return query, rag_questions

    def exec(self, inputs):
        query, rag_questions = inputs
        logger.info("📚 [OQARetrieve] EXEC - Starting sequential OQA retrieval: user query first, then RAG")
        
        queries = []
        # if query:
        #     queries.append(query)
        if rag_questions:
            queries.extend([q for q in rag_questions if q])

        aggregated = []
        best_seen = 0.0
        for i, q in enumerate(queries):
            logger.info(f"📚 [OQARetrieve] EXEC - Query {i+1}/{len(queries)}: '{q[:50]}...'")
            res, sc = retrieve_oqa(q, top_k=5)
            logger.info(f"📚 [OQARetrieve] EXEC - Retrieved {len(res) if res else 0} results, best score: {sc:.4f}")
            if res:
                aggregated.extend(res)
                if sc and sc > best_seen:
                    best_seen = sc

        # deduplicate by id or question
        seen = {}
        def key(it):
            return it.get("id") or it.get("question", "").lower()

        for it in aggregated:
            k = key(it)
            if not k:
                continue
            cur = seen.get(k)
            if cur is None or float(it.get("score", 0.0)) > float(cur.get("score", 0.0)):
                seen[k] = it
        uniq = list(seen.values())
        uniq.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        top5 = uniq[:5]
        
        # Log top results with more detail (id/topic only)
        if top5:
            lines = ["\n🏷️ [OQARetrieve] TOP-5 SCORES (desc):"]
            for i, it in enumerate(top5, 1):
                q = str(it.get('question', ''))
                sc = float(it.get('score', 0.0))
                topic = str(it.get('topic', ''))
                src_id = str(it.get('id', ''))
                lines.append(f"  {i}. score={sc:.4f} | Q: {q[:80]}... | Topic: {topic[:40]} | Id: {src_id}")
            logger.info("\n".join(lines))
            
            # Log aggregation statistics
            score_range = f"{top5[-1].get('score', 0.0):.4f} - {top5[0].get('score', 0.0):.4f}"
            logger.info(f"📊 [OQARetrieve] SCORE RANGE: {score_range}, Processed {len(aggregated)} total items before dedup")
        
        final_score = top5[0].get("score", 0.0) if top5 else 0.0
        logger.info(f"📚 [OQARetrieve] EXEC - Final aggregated: {len(top5)} unique results, top_score={final_score:.4f}")
        return top5, final_score

    def post(self, shared, prep_res, exec_res):
        logger.info("📚 [OQARetrieve] POST - Saving OQA retrieval results")
        items, score = exec_res
        # map to existing formatting function by adapting keys
        # store rich objects in separate key
        shared["oqa_hits"] = items
        # Build VN-style Q&A list string from English items
        hits_for_prompt = []
        for it in items:
            hits_for_prompt.append({
                "cau_hoi": it.get("question", ""),
                "cau_tra_loi": it.get("context", ""),
                "score": it.get("score", 0.0),
            })
        shared["retrieved"] = hits_for_prompt
        shared["retrieval_score"] = float(score)
        shared["need_clarify"] = float(score) < get_score_threshold()
        
        logger.info(f"📚 [OQARetrieve] POST - Saved {len(items)} OQA results, score: {score:.4f}, need_clarify: {shared['need_clarify']}")
        return "default"


class OQAComposeAnswerVIWithSources(Node):
    def prep(self, shared):
        logger.info("✍️ [OQACompose] PREP - Building Vietnamese composition with sources")
        role = shared.get("role", "orthodontist")
        query = shared.get("query", "")
        conversation_history = shared.get("conversation_history", [])
        persona = PERSONA_BY_ROLE.get(role, PERSONA_BY_ROLE.get("patient_diabetes"))
        ai_role = persona.get("persona", "Bác sĩ nha khoa")
        audience = persona.get("audience", "bác sĩ nha khoa")
        tone = persona.get("tone", "ngắn gọn, chính xác")
        # compact English QA block with topic and id sources
        items = shared.get("oqa_hits", [])
        lines = []
        for it in items:
            q = it.get("question", "")
            ctx = it.get("context", "")
            topic = it.get("topic", "")
            src_id = it.get("id", "")
            lines.append(f"Topic: {topic}\nQ: {q}\nContext: {ctx}\nSourceId: {src_id}\n")
        relevant_info = "\n".join(lines) if lines else "(no retrieved info)"
        formatted_history = format_conversation_history(conversation_history)
        prompt = PROMPT_OQA_COMPOSE_VI_WITH_SOURCES.format(
            ai_role=ai_role,
            audience=audience,
            tone=tone,
            query=query,
            relevant_info_from_kb=relevant_info,
            conversation_history=formatted_history,
        )
        logger.info(f"✍️ [OQACompose] PREP - Role: {role}, Query: '{query[:50]}...', OQA sources: {len(items)}")
        return prompt

    def exec(self, prompt):
        logger.info("✍️ [OQACompose] EXEC - Calling LLM for Vietnamese composition with sources")
        # Log the exact prompt being sent to LLM
        try:
            logger.info("✍️ [OQACompose] PROMPT (len=%d):\n%s", len(prompt) if isinstance(prompt, str) else 0, prompt)
        except Exception:
            pass
        try:
            resp = call_llm(prompt, max_retry_time=timeout_config.LLM_RETRY_TIMEOUT)
            logger.info(f"✍️ [OQACompose] EXEC - Raw LLM response length: {len(resp)} chars")
            logger.info(f"✍️ [OQACompose] EXEC - Full API response:\n{resp}")
            
            # First try normal parsing
            result = parse_yaml_with_schema(
                resp,
                required_fields=["explanation", "reference_ids", "suggestion_questions"],
                field_types={"explanation": str, "reference_ids": list, "suggestion_questions": list},
            )
            
            if not result or isinstance(result, str):
                logger.warning("✍️ [OQACompose] EXEC - Standard YAML parsing failed, trying manual extraction")
                # Manual fallback parsing for debugging
                try:
                    import re
                    import yaml
                    
                    # Extract YAML from code fences
                    yaml_match = re.search(r'```(?:yaml)?\s*\n(.*?)\n```', resp, re.DOTALL | re.IGNORECASE)
                    if yaml_match:
                        yaml_content = yaml_match.group(1).strip()
                        logger.info(f"✍️ [OQACompose] EXEC - Extracted YAML content:\n{yaml_content}")
                        
                        # Try to parse extracted YAML
                        parsed = yaml.safe_load(yaml_content)
                        if isinstance(parsed, dict):
                            result = {
                                "explanation": parsed.get("explanation", ""),
                                "reference_ids": parsed.get("reference_ids", []),
                                "suggestion_questions": parsed.get("suggestion_questions", [])
                            }
                            logger.info("✍️ [OQACompose] EXEC - Manual YAML parsing successful")
                        else:
                            logger.warning(f"✍️ [OQACompose] EXEC - YAML parsed but not dict: {type(parsed)}")
                    else:
                        logger.warning("✍️ [OQACompose] EXEC - No YAML code fence found in response")
                        
                except Exception as e:
                    logger.error(f"✍️ [OQACompose] EXEC - Manual parsing also failed: {e}")
            
            if not result or isinstance(result, str):
                logger.warning("✍️ [OQACompose] EXEC - All parsing failed, using fallback")
                return {
                    "explain": "Xin lỗi, tôi chưa thể tổng hợp câu trả lời phù hợp lúc này.",
                    "reference_ids": [],
                    "suggestion_questions": [],
                    "preformatted": True,
                }
            
            # Get full references from KB using IDs
            reference_ids = result.get("reference_ids", [])
            logger.info(f"✍️ [OQACompose] EXEC - Got {len(reference_ids)} reference IDs: {reference_ids}")
            
            # Query KB for full references
            id_to_ref = get_references_by_ids(reference_ids)
            logger.info(f"✍️ [OQACompose] EXEC - Retrieved {len(id_to_ref)} full references from KB")
            
            # Build sources list with required format: [N] TITLE LINK
            sources = format_references_numbered(reference_ids, id_to_ref)
            missing_ids = [rid for rid in reference_ids if rid not in id_to_ref]
            for rid in missing_ids:
                logger.warning(f"✍️ [OQACompose] EXEC - ID not found in KB: {rid}")
            
            # Append sources to explanation
            explanation = result.get("explanation", "")
            if sources:
                explanation += "\n\n**Nguồn tham khảo:**\n" + "\n".join(sources)
                logger.info(f"✍️ [OQACompose] EXEC - Appended {len(sources)} sources to explanation")
            
            logger.info(f"✍️ [OQACompose] EXEC - Successful composition: explanation_len={len(explanation)}, sources={len(sources)}, suggestions={len(result.get('suggestion_questions', []))}")
            return {
                "explain": explanation,
                "sources": sources,
                "suggestion_questions": result.get("suggestion_questions", []),
                "preformatted": True,
            }
            
        except APIOverloadException:
            logger.warning("✍️ [OQACompose] EXEC - API overloaded, using fallback response")
            return {
                "explain": "Xin lỗi, dịch vụ hiện đang quá tải. Vui lòng thử lại sau.",
                "reference_ids": [],
                "suggestion_questions": ["Try asking a simpler question", "Check back later", "Rephrase your question"],
                "preformatted": True,
            }

    def post(self, shared, prep_res, exec_res):
        logger.info("✍️ [OQACompose] POST - Saving composition results")
        shared["answer_obj"] = exec_res
        shared["explain"] = exec_res.get("explain", "")
        shared["sources"] = exec_res.get("sources", [])
        shared["suggestion_questions"] = exec_res.get("suggestion_questions", [])
        logger.info(f"✍️ [OQACompose] POST - Saved: explanation preview: '{exec_res.get('explain', '')[:100]}...', sources: {len(exec_res.get('sources', []))}")
        return "default"


class OQAClarify(Node):
    def prep(self, shared):
        logger.info("❓ [OQAClarify] PREP - Preparing clarification for low-score OQA query")
        role = shared.get("role", "orthodontist")
        items = shared.get("oqa_hits", [])
        score = shared.get("retrieval_score", 0.0)
        logger.info(f"❓ [OQAClarify] PREP - Role: {role}, Retrieved items: {len(items)}, Score: {score:.4f}")
        return role, items

    def exec(self, inputs):
        role, items = inputs
        logger.info("❓ [OQAClarify] EXEC - Generating clarification for low-score OQA result")
        
        if not items:
            logger.info("❓ [OQAClarify] EXEC - No retrieved items, getting random OQA suggestions")
            random_items = retrieve_random_oqa(amount=5)
            suggestions = [it.get("question", "") for it in random_items if it.get("question")]
            logger.info(f"❓ [OQAClarify] EXEC - Generated {len(suggestions)} random suggestions")
        else:
            logger.info("❓ [OQAClarify] EXEC - Using retrieved items for suggestions")
            suggestions = [it.get("question", "") for it in items[:5] if it.get("question")]
            logger.info(f"❓ [OQAClarify] EXEC - Generated {len(suggestions)} suggestions from retrieved items")
        
        return {
            "explain": "Hiện tại dữ liệu retrieve từ OQA chưa đủ liên quan. Bạn có thể chọn/diễn đạt lại từ các gợi ý sau (bằng tiếng Việt).",
            "suggestion_questions": suggestions,
            "preformatted": True,
        }

    def post(self, shared, prep_res, exec_res):
        logger.info("❓ [OQAClarify] POST - Saving clarification response")
        shared["answer_obj"] = exec_res
        shared["explain"] = exec_res.get("explain", "")
        shared["suggestion_questions"] = exec_res.get("suggestion_questions", [])
        logger.info(f"❓ [OQAClarify] POST - Saved clarification with {len(exec_res.get('suggestion_questions', []))} suggestions")
        return "default"


class OQAChitChat(Node):
    """Specialized chitchat node for OQA/orthodontist context using LLM."""
    
    def prep(self, shared):
        logger.info("💬 [OQAChitChat] PREP - Preparing orthodontic chitchat response")
        role = shared.get("role", "orthodontist")
        query = shared.get("query", "")
        conversation_history = shared.get("conversation_history", [])
        logger.info(f"💬 [OQAChitChat] PREP - Role: {role}, Query: '{query[:50]}...', History: {len(conversation_history)} messages")
        return role, query, conversation_history
    
    def exec(self, inputs):
        role, query, conversation_history = inputs
        logger.info("💬 [OQAChitChat] EXEC - Calling LLM for orthodontic chitchat")
        
        # Format conversation history like ChitChatRespond
        history_lines = []
        for msg in conversation_history[-6:]:  # Last 3 pairs (6 messages)
            try:
                who = msg.get("role")
                content = msg.get("content", "")
                history_lines.append(f"- {who}: {content}")
            except Exception:
                continue
        formatted_history = "\n".join(history_lines)
        logger.info(f"💬 [OQAChitChat] EXEC - Formatted {len(history_lines)} conversation context lines")
        
        # Get persona for orthodontist role (safe fallback)
        if role in PERSONA_BY_ROLE:
            persona = PERSONA_BY_ROLE[role]
            audience = persona.get('audience', 'bác sĩ nha khoa')
            tone = persona.get('tone', 'chuyên nghiệp, súc tích')
        else:
            audience, tone = 'bác sĩ nha khoa', 'chuyên nghiệp, súc tích'
        
        # Build prompt for OQA chitchat
        prompt = PROMPT_OQA_CHITCHAT.format(
            conversation_history=formatted_history,
            query=query,
            role=role,
            audience=audience,
            tone=tone
        )
        
        try:
            # Log the exact prompt being sent to LLM
            try:
                logger.info("💬 [OQAChitChat] PROMPT (len=%d):\n%s", len(prompt) if isinstance(prompt, str) else 0, prompt)
            except Exception:
                pass
            resp = call_llm(prompt)
            logger.info(f"💬 [OQAChitChat] EXEC - Raw chitchat response length: {len(resp)} chars")
            logger.info(f"💬 [OQAChitChat] EXEC - Full API response:\n{resp}")
            
            # Generate orthodontic-related suggestions
            suggestions = [
                "Các phương pháp retention hiện đại nhất là gì?",
                "Làm thế nào xử lý root resorption?", 
                "Best practices cho bracket placement?",
                "Clear aligner treatment như thế nào?",
                "Cách xử lý các case khó?"
            ]
            
            logger.info(f"💬 [OQAChitChat] EXEC - Generated response with {len(suggestions[:3])} orthodontic suggestions")
            return {
                "explain": resp,
                "suggestion_questions": suggestions[:3],
                "preformatted": True
            }
            
        except APIOverloadException:
            logger.warning("💬 [OQAChitChat] EXEC - API overloaded, using fallback chitchat")
            # Fallback response without calling external fallback node
            fallback_resp = "Xin chào! Tôi là trợ lý chuyên môn nha khoa chỉnh nha. Hiện tại hệ thống đang quá tải, nhưng tôi sẵn sàng hỗ trợ bạn!"
            suggestions = [
                "Hỏi về retention protocols",
                "Tìm hiểu về bracket placement", 
                "Thảo luận về clear aligners"
            ]
            return {
                "explain": fallback_resp,
                "suggestion_questions": suggestions,
                "preformatted": True
            }
    
    def post(self, shared, prep_res, exec_res):
        logger.info("💬 [OQAChitChat] POST - Saving chitchat response")
        shared["answer_obj"] = exec_res
        shared["explain"] = exec_res.get("explain", "")
        shared["suggestion_questions"] = exec_res.get("suggestion_questions", [])
        logger.info(f"💬 [OQAChitChat] POST - Saved chitchat: '{exec_res.get('explain', '')[:50]}...', suggestions: {len(exec_res.get('suggestion_questions', []))}")
        return "default"