from typing import List, Optional, Any
from qdrant_client import models
import logging

logger = logging.getLogger(__name__)

class FilterBuilder:
    def __init__(self):
        self.conditions = []

    def add_match(self, key: str, value: Any):
        """Adds a match condition (exact match)."""
        if value:
            self.conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
            )
        return self

    def add_demuc(self, demuc: Optional[str]):
        return self.add_match("DEMUC", demuc)

    def add_subtopic(self, subtopic: Optional[str]):
        return self.add_match("CHUDECON", subtopic)

    def add_user_id(self, user_id: Optional[str]):
        return self.add_match("user_id", user_id)

    def build(self) -> Optional[models.Filter]:
        """Returns the constructed Filter object or None if no conditions."""
        if not self.conditions:
            return None
        return models.Filter(must=self.conditions)
