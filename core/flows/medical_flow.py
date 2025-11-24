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

def create_retrieve_flow(fallback_node):
    """
    Create a reusable retrieval sub-flow: topic_classify → retrieve_kb

    This flow handles:
    1. Topic classification (DEMUC only)
    2. Retrieval from knowledge base

    Args:
        fallback_node: The parent flow's fallback node to route errors to

    Returns:
        Flow: A flow that starts with topic_classify
    """
    from ..nodes import   (
         RetrieveFromKB, TopicClassifyAgent
    )
    
    from pocketflow import Flow 
    logger.info("[retrieve_flow] Creating retrieval sub-flow")

    # Create retrieval pipeline nodes
    topic_classify = TopicClassifyAgent(max_retries=3,wait=2)
    retrieve_kb = RetrieveFromKB()
    
    topic_classify >> retrieve_kb  
    retrieve_kb - "fallback" >> fallback_node

    retrieve_flow = Flow(start=topic_classify)
    logger.info("[retrieve_flow] Retrieval sub-flow created: topic_classify → retrieve_kb")
    return retrieve_flow


def create_med_agent_flow():
    from ..nodes import (
        IngestQuery, DecideSummarizeConversationToRetriveOrDirectlyAnswer, RagAgent, ComposeAnswer,
        FallbackNode,QueryCreatingForRetrievalAgent
    )
    from pocketflow import Flow 
    logger.info("[Flow] Tạo medical agent flow với retrieve_flow sub-flow")

    # Create nodes
    ingest = IngestQuery()
    main_decision = DecideSummarizeConversationToRetriveOrDirectlyAnswer()
    fallback = FallbackNode()
    rag_agent = RagAgent(max_retries=2)
    compose_answer = ComposeAnswer()
    better_retrieval_query = QueryCreatingForRetrievalAgent()
    # Create retrieve_flow sub-flow (pass fallback node for error routing)
    retrieve_flow = create_retrieve_flow(fallback)

   
    logger.info("[Flow] Kết nối nodes: Ingest → MainDecision → [STOP if direct_response OR retrieve_flow → RagAgent → ComposeAnswer]")

    ingest >> main_decision
    # Step 2: From MainDecision
    main_decision - "retrieve_kb" >> rag_agent
    # Note: "direct_response" action has NO connection → flow ends, answer already in shared

    rag_agent - "create_retrieval_query" >> better_retrieval_query  
    better_retrieval_query >> retrieve_flow
    rag_agent - "retrieve_kb" >> retrieve_flow  # Loop back for more retrieval
    retrieve_flow >> rag_agent
    
    rag_agent - "compose_answer" >> compose_answer
    
    
    # Fallback  
    main_decision - "fallback" >> fallback
    rag_agent - "fallback" >> fallback
    compose_answer - "fallback" >> fallback
    better_retrieval_query - "fallback" >> fallback

    flow = Flow(start=ingest)
    logger.info("[Flow] Medical agent flow với retrieve_flow sub-flow đã được tạo thành công")
    return flow



def create_oqa_orthodontist_flow():
    from pocketflow import Flow 
    
    return Flow(start=None)


    