import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import re
from unidecode import unidecode
from rank_bm25 import BM25Okapi

# Optional dependencies (kept for backward compatibility, not used in BM25 flow)
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


OQA_CSV_PATH = os.path.join("medical_knowledge_base", "oqa_v1_dataset.csv")


def _ensure_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    return " ".join(s.replace("\n", " ").replace("\r", " ").split())


class OQAVectorIndex:
    """In-memory BM25 index for OQA English dataset.

    - Uses BM25Okapi over tokenized `question + context + topic`.
    - Returns only fields: question, context, topic, id (no answers/reference).
    """

    def __init__(self, csv_path: str = OQA_CSV_PATH) -> None:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"OQA dataset not found: {csv_path}")

        # Load
        df = None
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                break
            except Exception:
                continue
        if df is None:
            raise ValueError(f"Failed to read OQA CSV: {csv_path}")

        # Normalize and keep only needed columns
        expected_cols = [
            "question",
            "context",
            "answers",
            "answer_sentence",
            "topic",
            "reference",
            "id",
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ""

        df = df[expected_cols].copy()
        for col in expected_cols:
            df[col] = df[col].apply(_ensure_str)

        # Build BM25 corpus from question + context + topic
        def _tokenize(text: str) -> List[str]:
            s = unidecode(_ensure_str(text)).lower()
            s = re.sub(r"[^a-z0-9\s]", " ", s)
            toks = [t for t in s.split() if t]
            return toks

        docs: List[str] = []
        for _, row in df.iterrows():
            q = row.get("question", "")
            ctx = row.get("context", "")
            topic = row.get("topic", "")
            combined = f"{q}\n{ctx}\n{topic}"
            docs.append(combined)

        tokenized_corpus: List[List[str]] = [_tokenize(t) for t in docs]
        self._bm25 = BM25Okapi(tokenized_corpus)

        # Store rows for result mapping
        self._df = df.reset_index(drop=True)

    def _tokenize_query(self, query: str) -> List[str]:
        s = unidecode(_ensure_str(query)).lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        return [t for t in s.split() if t]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query or len(self._df) == 0:
            return []
        q_tokens = self._tokenize_query(query)
        scores = self._bm25.get_scores(q_tokens)
        scores = np.array(scores, dtype=np.float32)
        k = int(min(top_k, scores.shape[0]))
        part = np.argpartition(scores, -k)[-k:]
        idxs = part[np.argsort(scores[part])[::-1]].tolist()

        results: List[Dict[str, Any]] = []
        for idx in idxs:
            row = self._df.iloc[int(idx)]
            sc = float(scores[int(idx)])
            results.append(
                {
                    "score": sc,
                    "question": _ensure_str(row.get("question", "")),
                    "context": _ensure_str(row.get("context", "")),
                    "topic": _ensure_str(row.get("topic", "")),
                    "id": _ensure_str(row.get("id", "")),
                }
            )
        return results

    def get_random(self, amount: int = 5) -> List[Dict[str, Any]]:
        if len(self._df) == 0:
            return []
        n = min(amount, len(self._df))
        sampled = self._df.sample(n=n, random_state=123)
        out: List[Dict[str, Any]] = []
        for _, row in sampled.iterrows():
            out.append(
                {
                    "score": 1.0,
                    "question": _ensure_str(row.get("question", "")),
                    "context": _ensure_str(row.get("context", "")),
                    "topic": _ensure_str(row.get("topic", "")),
                    "id": _ensure_str(row.get("id", "")),
                }
            )
        return out


_OQA_INDEX: Optional[OQAVectorIndex] = None


def get_oqa_index() -> OQAVectorIndex:
    global _OQA_INDEX
    if _OQA_INDEX is None:
        _OQA_INDEX = OQAVectorIndex()
    return _OQA_INDEX


def preload_oqa_index() -> None:
    """Preload OQA index into memory during server startup."""
    global _OQA_INDEX
    if _OQA_INDEX is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(" Loading OQA vector index into memory...")
        try:
            _OQA_INDEX = OQAVectorIndex()
            logger.info(f" OQA index loaded successfully: {len(_OQA_INDEX._df)} items using BM25")
        except Exception as e:
            logger.error(f" Failed to load OQA index: {e}")
            raise


def is_oqa_index_loaded() -> bool:
    """Check if OQA index is already loaded in memory."""
    return _OQA_INDEX is not None


def retrieve_oqa(query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], float]:
    idx = get_oqa_index()
    res = idx.search(query, top_k=top_k)
    score = float(res[0]["score"]) if res else 0.0
    return res, score


def retrieve_random_oqa(amount: int = 5) -> List[Dict[str, Any]]:
    idx = get_oqa_index()
    return idx.get_random(amount)


