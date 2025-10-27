from pocketflow import Flow
from core.nodes.medical_nodes import (
    IngestQuery, MainDecisionAgent, ScoreDecisionNode, RetrieveFromKB,
    ComposeAnswer, ClarifyQuestionNode, GreetingResponse, FallbackNode, ChitChatRespond,
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
    logger.info("[Flow] Tạo medical agent flow với modular architecture")
    
    # Create nodes
    ingest = IngestQuery()
    main_decision = MainDecisionAgent()
    retrieve_kb = RetrieveFromKB()
    score_decision = ScoreDecisionNode()
    compose_answer = ComposeAnswer()
    clarify_question = ClarifyQuestionNode()
    greeting = GreetingResponse()
    fallback = FallbackNode()
    chitchat = ChitChatRespond()
    logger.info("[Flow] Kết nối nodes theo luồng mới")
    
    # Main flow: Ingest → MainDecision
    ingest >> main_decision
    
    # From MainDecision, route based on classification
    main_decision - "retrieve_kb" >> retrieve_kb
    # main_decision no longer routes directly to topic_suggest
    main_decision - "greeting" >> greeting
    main_decision - "chitchat" >> chitchat
    main_decision - "fallback" >> fallback
    # From RetrieveKB, check score and decide
    retrieve_kb >> score_decision
    
    # From ScoreDecision, route to appropriate action
    score_decision - "compose_answer" >> compose_answer
    score_decision - "clarify" >> clarify_question

    # ComposeAnswer is terminal by default, but can route to fallback if API overloaded
    compose_answer - "fallback" >> fallback
    # compose_answer with "default" action is terminal (no routing needed)

    # ChitChat can route to fallback if API overloaded
    chitchat - "fallback" >> fallback
    # chitchat with "default" action is terminal

    # ClarifyQuestion is terminal (no routing needed)
    
   
    flow = Flow(start=ingest)
    logger.info("[Flow] Medical agent flow với modular architecture đã được tạo thành công")
    return flow



def create_oqa_orthodontist_flow():
    logger.info("[Flow] Tạo OQA orthodontist flow với nodes độc lập")

    # Create independent OQA nodes (no reuse from old flow)
    ingest = OQAIngestDefaults()
    classify = OQAClassifyEN()
    retrieve = OQARetrieve()
    score = ScoreDecisionNode()  # Only reuse ScoreDecisionNode as it's generic
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

