"""Node definitions for medical chatbot flows.

Refactored: original monolithic `medical_nodes.py` has been split so each node
now lives in its own module (e.g. `IngestQuery.py`). This file re-exports those
classes for convenient imports like `from core.nodes import IngestQuery`.
Add future nodes (e.g. GreetingResponse) by importing and appending to `__all__`.
"""

# Medical node class imports (split from previous medical_nodes.py)
from .IngestQuery import IngestQuery
from .DecideSummarizeConversationToRetriveOrDirectlyAnswer import DecideSummarizeConversationToRetriveOrDirectlyAnswer
from .RetrieveFromKB import RetrieveFromKB
from .ComposeAnswer import ComposeAnswer
from .FallbackNode import FallbackNode
from .RagAgent import RagAgent
from .QueryExpandAgent import QueryExpandAgent
from .TopicClassifyAgent import TopicClassifyAgent
from .QueryCreatingForRetrievalAgent import QueryCreatingForRetrievalAgent 



# Future / optional nodes:
# from .GreetingResponse import GreetingResponse  # Uncomment when implemented

# OQA nodes remain commented until re-enabled after refactor.
# from .oqa_nodes import (
#     OQAIngestDefaults,
#     OQAClassifyEN,
#     OQARetrieve,
#     OQAComposeAnswerVIWithSources,
#     OQAClarify,
#     OQAChitChat,
# )

__all__ = [
    # Medical nodes
    "IngestQuery",
    "DecideSummarizeConversationToRetriveOrDirectlyAnswer",
    "RetrieveFromKB",
    "ComposeAnswer",
    "FallbackNode",
    "RagAgent",
    "QueryExpandAgent",
    "TopicClassifyAgent",
    "QueryCreatingForRetrievalAgent",
    # "GreetingResponse",  # add when implemented
    # OQA nodes
    # "OQAIngestDefaults",
    # "OQAClassifyEN",
    # "OQARetrieve",
    # "OQAComposeAnswerVIWithSources",
    # "OQAClarify",
    # "OQAChitChat",
]
