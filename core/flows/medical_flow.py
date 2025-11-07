from pocketflow import Flow
from core.nodes.medical_nodes import (
    IngestQuery, MainDecisionAgent, TopicClassifyAgent, QueryExpandAgent, 
    RagAgent, RetrieveFromKB, ComposeAnswer, GreetingResponse, 
    FallbackNode, ChitChatRespond,
)
from core.nodes.oqa_nodes import (
    OQAIngestDefaults, OQAClassifyEN, OQARetrieve,
    OQAComposeAnswerVIWithSources, OQAClarify, OQAChitChat
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

def create_med_agent_flow():
    logger.info("[Flow] Tạo medical agent flow với RagAgent làm decision maker chính")

    # Create nodes
    ingest = IngestQuery()
    main_decision = MainDecisionAgent()
    rag_agent = RagAgent()
    topic_classify = TopicClassifyAgent()
    query_expand = QueryExpandAgent()
    retrieve_kb = RetrieveFromKB()
    compose_answer = ComposeAnswer()
    fallback = FallbackNode()
    chitchat = ChitChatRespond()
    
    logger.info("[Flow] Kết nối nodes theo flow mới: Ingest → MainDecision → RagAgent (orchestrator)")

    # Step 1: Ingest → MainDecision
    ingest >> main_decision

    # Step 2: From MainDecision - route to RagAgent or Chitchat
    main_decision - "retrieve_kb" >> rag_agent
    main_decision - "chitchat" >> chitchat
    main_decision - "fallback" >> fallback

    # Step 3: RagAgent orchestrates the pipeline
    # RagAgent can route to: classify, expand, retrieve, or compose_answer
    
    # Route to TopicClassifyAgent for metadata extraction
    rag_agent - "classify" >> topic_classify
    topic_classify >> rag_agent  # Route back to RagAgent
    topic_classify - "fallback" >> fallback
    
    # Route to QueryExpandAgent for query expansion
    rag_agent - "expand" >> query_expand
    query_expand >> rag_agent  # Route back to RagAgent
    query_expand - "fallback" >> fallback
    
    # Route to RetrieveFromKB for data retrieval
    rag_agent - "retrieve" >> retrieve_kb
    retrieve_kb >> rag_agent  # Route back to RagAgent
    
    # Route to ComposeAnswer when ready
    rag_agent - "compose_answer" >> compose_answer
    compose_answer - "fallback" >> fallback
    # compose_answer with "default" action is terminal

    # ChitChat can route to fallback if API overloaded
    chitchat - "fallback" >> fallback
    # chitchat with "default" action is terminal

    flow = Flow(start=ingest)
    logger.info("[Flow] Medical agent flow mới với RagAgent orchestrator đã được tạo thành công")
    return flow



def create_oqa_orthodontist_flow():
    logger.info("[Flow] Tạo OQA orthodontist flow với nodes độc lập")

    # Create independent OQA nodes (no reuse from old flow)
    ingest = OQAIngestDefaults()
    classify = OQAClassifyEN()
    retrieve = OQARetrieve()
    compose = OQAComposeAnswerVIWithSources()
    clarify = OQAClarify()
    chitchat = OQAChitChat()  # OQA-specific chitchat

    # Wire the flow (no fallback node)
    ingest >> classify
    classify - "retrieve_kb" >> retrieve
    classify - "chitchat" >> chitchat

    retrieve >> score
    score - "compose_answer" >> compose
    score - "clarify" >> clarify

    # All terminal nodes (no fallback routing)
    
    flow = Flow(start=ingest)
    logger.info("[Flow] OQA orthodontist flow với nodes độc lập đã được tạo thành công")
    return flow

