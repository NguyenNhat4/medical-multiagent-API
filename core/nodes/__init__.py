"""Node definitions for medical chatbot flows.

Refactored: original monolithic `medical_nodes.py` has been split so each node
now lives in its own module (e.g. `IngestQuery.py`). This file re-exports those
classes for convenient imports like `from core.nodes import IngestQuery`.
Add future nodes (e.g. GreetingResponse) by importing and appending to `__all__`.
"""

# Medical node class imports (split from previous medical_nodes.py)
from .IngestQuery import IngestQuery
from .DecideToRetriveOrAnswer import DecideToRetriveOrAnswer
from .RetrieveFromKB import RetrieveFromKB
from .ComposeAnswer import ComposeAnswer
from .FallbackNode import FallbackNode
from .FilterAgent import FilterAgent
from .RagAgent import RagAgent
from .QueryExpandAgent import QueryExpandAgent
from .TopicClassifyAgent import TopicClassifyAgent 



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
    "DecideToRetriveOrAnswer",
    "RetrieveFromKB",
    "ComposeAnswer",
    "FallbackNode",
    "FilterAgent",
    "RagAgent",
    "QueryExpandAgent",
    "TopicClassifyAgent",
    # "GreetingResponse",  # add when implemented
    # OQA nodes
    # "OQAIngestDefaults",
    # "OQAClassifyEN",
    # "OQARetrieve",
    # "OQAComposeAnswerVIWithSources",
    # "OQAClarify",
    # "OQAChitChat",
]
