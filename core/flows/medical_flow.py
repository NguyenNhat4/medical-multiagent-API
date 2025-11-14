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
    Create a reusable retrieval sub-flow: topic_classify → retrieve_kb → filter_agent

    This flow handles:
    1. Topic classification (DEMUC only)
    2. Retrieval from knowledge base
    3. Filtering candidates

    Args:
        fallback_node: The parent flow's fallback node to route errors to

    Returns:
        Flow: A flow that starts with topic_classify
    """
    from ..nodes import   (
        TopicClassifyAgent, RetrieveFromKB, FilterAgent
    )
    
    from pocketflow import Flow 
    logger.info("[retrieve_flow] Creating retrieval sub-flow")

    # Create retrieval pipeline nodes
    topic_classify = TopicClassifyAgent()
    retrieve_kb = RetrieveFromKB()
    filter_agent = FilterAgent()

    # Connect retrieval pipeline
    # topic_classify classifies DEMUC only, then proceeds to retrieval
    topic_classify >> retrieve_kb
    topic_classify - "fallback" >> fallback_node

    # retrieve_kb → filter_agent
    retrieve_kb >> filter_agent
    retrieve_kb - "fallback" >> fallback_node

    # filter_agent → default (will return to parent flow)
    filter_agent - "fallback" >> fallback_node

    retrieve_flow = Flow(start=topic_classify)
    logger.info("[retrieve_flow] Retrieval sub-flow created: topic_classify → retrieve_kb → filter_agent")
    return retrieve_flow


def create_med_agent_flow():
    from ..nodes import (
        IngestQuery, DecideToRetriveOrAnswer, RagAgent, ComposeAnswer,
        FallbackNode,
    )
    from pocketflow import Flow 
    logger.info("[Flow] Tạo medical agent flow với retrieve_flow sub-flow")

    # Create nodes
    ingest = IngestQuery()
    main_decision = DecideToRetriveOrAnswer()
    fallback = FallbackNode()

    # Create retrieve_flow sub-flow (pass fallback node for error routing)
    retrieve_flow = create_retrieve_flow(fallback)

    # Create decision and answer nodes
    rag_agent = RagAgent()
    compose_answer = ComposeAnswer()

    logger.info("[Flow] Kết nối nodes: Ingest → MainDecision → [STOP if direct_response OR retrieve_flow → RagAgent → ComposeAnswer]")

    # Step 1: Ingest → MainDecision
    ingest >> main_decision

    # Step 2: From MainDecision
    # - direct_response: Answer already saved in shared by MainDecision, flow stops (no next node)
    # - retrieve_kb: Continue to retrieve_flow
    main_decision - "retrieve_kb" >> retrieve_flow
    main_decision - "fallback" >> fallback
    # Note: "direct_response" action has NO connection → flow ends, answer already in shared

    # Step 3: retrieve_flow → rag_agent (rag_agent decides next step)
    retrieve_flow >> rag_agent

    # Step 4: RagAgent routing
    rag_agent - "retry_retrieve" >> retrieve_flow  # Loop back for more retrieval
    rag_agent - "compose_answer" >> compose_answer
    rag_agent - "fallback" >> fallback

    # compose_answer (terminal)
    compose_answer - "fallback" >> fallback

    flow = Flow(start=ingest)
    logger.info("[Flow] Medical agent flow với retrieve_flow sub-flow đã được tạo thành công")
    return flow



def create_oqa_orthodontist_flow():
    from pocketflow import Flow 
    
    return Flow(start=None)