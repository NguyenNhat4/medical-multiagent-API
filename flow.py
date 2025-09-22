from pocketflow import Flow
from nodes import (
    IngestQuery, MainDecisionAgent, ScoreDecisionNode, RetrieveFromKB, 
    ComposeAnswer, ClarifyQuestionNode, GreetingResponse, FallbackNode,
    ChitChatRespond,
)
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

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
    
    # From ComposeAnswer, route to fallback if API overloaded
    compose_answer - "fallback" >> fallback
    
    # ChitChatRespond is terminal for non-RAG cases
    
   
    flow = Flow(start=ingest)
    logger.info("[Flow] Medical agent flow với modular architecture đã được tạo thành công")
    return flow


