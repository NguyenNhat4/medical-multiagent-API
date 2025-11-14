# Core framework import
from pocketflow import Node

# Standard library imports
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



class FallbackNode(Node):
    """Node fallback khi API quá tải - retrieve query và trả kết quả dựa trên score"""
    
    def prep(self, shared):
        logger.info("🔄 [FallbackNode] PREP - Xử lý fallback khi API quá tải")
        query = shared.get("query", "")
        role = shared.get("role", "")
        return query, role

    def exec(self, inputs):
        # Import dependencies only when needed
        from unidecode import unidecode
        from utils.knowledge_base import get_kb, ROLE_TO_CSV, retrieve_random_by_role
        from utils.helpers import aggregate_retrievals, format_kb_qa_list

        query, role = inputs
        logger.info(f"🔄 [FallbackNode] EXEC - Fallback search cho role: {role} với query: '{query[:50]}...'")

        try:
            # 1) Tìm tuần tự trong CSV theo role, so khớp HOÀN TOÀN với cột CAUHOI
            kb = get_kb()
            role_lower = (role or "").lower()
            role_csv = ROLE_TO_CSV.get(role_lower)

            def _norm_text(s: str) -> str:
                s = unidecode((s or "").lower())
                return " ".join(s.split())

            q_norm = _norm_text(query)
            exact_matches = []

            if role_csv and role_csv in kb.role_dataframes:
                df = kb.role_dataframes[role_csv]
                for _, row in df.iterrows():
                    q_text = str(row.get("CAUHOI", ""))
                    a_text = str(row.get("CAUTRALOI", ""))
                    qn = _norm_text(q_text)
                    if qn and q_norm and qn == q_norm:
                        exact_matches.append({
                            "cau_hoi": q_text,
                            "cau_tra_loi": a_text,
                            "de_muc": row.get("DEMUC", ""),
                            "chu_de_con": row.get("CHUDECON", ""),
                            "ma_so": row.get("MASO", ""),
                            "keywords": row.get("keywords", ""),
                            "giai_thich": row.get("GIAITHICH", ""),
                        })

            # Build retrieval queries: use only the main query
            retrieval_queries = []
            if query:
                retrieval_queries.append(query)

            # Use aggregate_retrievals helper function
            retrieved_results, _ = aggregate_retrievals(retrieval_queries, role=role, top_k=15)

            try:
                formatted = format_kb_qa_list(retrieved_results, max_items=15)
                if formatted:
                    logger.info("\n📚 [FallbackNode] RETRIEVE - Aggregated Results:\n" + formatted)
            except Exception:
                pass

            # Log thêm bảng điểm cho retrieved_results
            if retrieved_results:
                lines = ["\n🏷️ [FallbackNode] TOP SCORES (desc):"]
                for i, it in enumerate(retrieved_results, 1):
                    q = str(it.get('cau_hoi', ''))
                    sc = float(it.get('score', 0.0))
                    lines.append(f"  {i}. score={sc:.4f} | Q: {q[:140]}")
                logger.info("\n".join(lines))

            if exact_matches:
                best = exact_matches[0]
                explain = best.get("cau_tra_loi", "")
                # Suggestions: top4 từ retrieve (khác câu exact match)
                suggestion_questions = []
                exact_q_norm = _norm_text(best.get("cau_hoi", ""))
                for it in retrieved_results:
                    q = it.get('cau_hoi', '')
                    if q and _norm_text(q) != exact_q_norm:
                        suggestion_questions.append(q)
                        if len(suggestion_questions) >= 4:
                            break
                score = 1.0
                # Log lựa chọn cuối
                logger.info("\n✅ [FallbackNode] EXPLAIN (exact match): score=1.0000 | Q (exact): " + str(best.get("cau_hoi", ""))[:140])
                if suggestion_questions:
                    # map score theo câu hỏi để log
                    score_map = {str(it.get('cau_hoi', '')): float(it.get('score', 0.0)) for it in retrieved_results}
                    sug_lines = ["📌 [FallbackNode] SUGGESTIONS (top4):"]
                    for idx, sq in enumerate(suggestion_questions, 1):
                        sug_lines.append(f"  {idx}. score={score_map.get(sq, 0.0):.4f} | Q: {sq[:140]}")
                    logger.info("\n".join(sug_lines))
            else:
                # Không có exact match: nếu có retrieved_results, dùng top1 làm explain và còn lại làm suggestion
                if retrieved_results:
                    best_answer = retrieved_results[0]
                    explain = best_answer.get("cau_tra_loi", "")
                    suggestion_questions = [it.get('cau_hoi', '') for it in retrieved_results[1:5] if it.get('cau_hoi')]
                    score = float(best_answer.get('score', 0.0))
                    # Log lựa chọn cuối
                    logger.info(f"\n✅ [FallbackNode] EXPLAIN (retrieve top1): score={score:.4f} | Q: {str(best_answer.get('cau_hoi',''))[:140]}")
                    if suggestion_questions:
                        sug_lines = ["📌 [FallbackNode] SUGGESTIONS (next4):"]
                        for idx, it in enumerate(retrieved_results[1:5], 1):
                            if not it.get('cau_hoi'):
                                continue
                            sug_lines.append(f"  {idx}. score={float(it.get('score', 0.0)):.4f} | Q: {str(it.get('cau_hoi'))[:140]}")
                        logger.info("\n".join(sug_lines))
                else:
                    explain = "Hiện tại tôi chưa có đủ thông tin liên quan để trả lời câu hỏi này của bạn, Bạn có thể đặt lại câu hỏi khác hoặc diễn đạt lại câu hỏi của bạn! Hoặc bạn có thể chọn các câu hỏi gợi ý dưới đây!"
                    random_questions = retrieve_random_by_role(role, amount=5)
                    suggestion_questions = [q['cau_hoi'] for q in random_questions]
                    score = 0.0

            result = {
                "explain": explain,
                "suggestion_questions": suggestion_questions,
                "retrieval_score": score,
                "preformatted": True
            }

            logger.info(f"🔄 [FallbackNode] EXEC - Generated response with {len(suggestion_questions)} suggestions")
            return result

        except Exception as e:
            logger.error(f"🔄 [FallbackNode] EXEC - Error during fallback: {e}")
            # Fallback tối thiểu
            return {
                "explain": "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau.",
                "suggestion_questions": [],
                "retrieval_score": 0.0,
                "preformatted": True
            }
    
    def post(self, shared, prep_res, exec_res):
        logger.info("🔄 [FallbackNode] POST - Lưu fallback response")
        shared["answer_obj"] = exec_res
        shared["explain"] = exec_res.get("explain", "")
        shared["suggestion_questions"] = exec_res.get("suggestion_questions", [])
        shared["retrieval_score"] = exec_res.get("retrieval_score", 0.0)
        return "default"

