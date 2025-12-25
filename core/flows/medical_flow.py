import logging
from pocketflow import Flow 

# Configure logging for this module with Vietnam timezone
from utils.timezone_utils import setup_vietnam_logging
from config.logging_config import logging_config
from tracing import trace_flow, TracingConfig
from ..nodes import (
    RetrieveFromKBWithDemuc,
    RetrieveFromKBWithoutDemuc,
    TopicClassifyAgent
)
from ..nodes import (
    IngestQuery, DecideSummarizeConversationToRetriveOrDirectlyAnswer, RagAgent, ComposeAnswer,
    FallbackNode,QueryCreatingForRetrievalAgent
)
# Import Memory nodes
from ..nodes.RetrieveFromMemory import RetrieveFromMemory
from ..nodes.SaveToMemory import SaveToMemory

if logging_config.USE_VIETNAM_TIMEZONE:
    logger = setup_vietnam_logging(__name__, 
                                 level=getattr(logging, logging_config.LOG_LEVEL.upper()),
                                 format_str=logging_config.LOG_FORMAT)
else:
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, logging_config.LOG_LEVEL.upper()))
    

@trace_flow(flow_name="MedFlow")
class MedFlow(Flow):
    def __init__(self):
        ingest = IngestQuery()
        retrieve_memory = RetrieveFromMemory()

        topic_classify = TopicClassifyAgent(max_retries=2)
        
        main_decision = DecideSummarizeConversationToRetriveOrDirectlyAnswer()
        fallback = FallbackNode()
        rag_agent = RagAgent(max_retries=2)
        compose_answer = ComposeAnswer()
        save_memory = SaveToMemory()
        better_retrieval_query = QueryCreatingForRetrievalAgent()
        
        retrieve_with_demuc = RetrieveFromKBWithDemuc()

        # Step 1: Ingest -> Memory Retrieval -> Main Decision
        ingest >> retrieve_memory >> main_decision
        
        # Step 2: From MainDecision
        main_decision - "retrieve_kb" >> rag_agent

        # Path 1: Retrieval with Demuc
        # Parallel Save: retrieve_with_demuc leads to BOTH save_memory AND compose_answer
        rag_agent - "create_retrieval_query" >> better_retrieval_query >> retrieve_with_demuc >> compose_answer
        retrieve_with_demuc >> save_memory # Parallel save path

        rag_agent - "retrieve_kb" >> topic_classify >> retrieve_with_demuc  # Loop back for more retrieval

        # Path 2: Direct Compose from RagAgent
        # Parallel Save: rag_agent leads to BOTH save_memory AND compose_answer
        rag_agent - "compose_answer" >> compose_answer
        rag_agent - "compose_answer" >> save_memory # Parallel save path
        
        # Fallback  
        main_decision - "fallback" >> fallback
        rag_agent - "fallback" >> fallback
        compose_answer - "fallback" >> fallback
        better_retrieval_query - "fallback" >> fallback
        
        super().__init__(start=ingest)


def create_oqa_orthodontist_flow():
    from pocketflow import Flow 
    return Flow(start=None)
