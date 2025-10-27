import os
from typing import Any, Dict, List, Optional, Tuple
import ast

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


def get_references_by_ids(ids: List[str]) -> Dict[str, str]:
    """Get full reference text for given list of IDs.
    
    Args:
        ids: List of document IDs
        
    Returns:
        Dict mapping ID to full reference text
    """
    idx = get_oqa_index()
    result = {}
    
    for doc_id in ids:
        # Find row with matching ID
        matching_rows = idx._df[idx._df["id"] == doc_id]
        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            result[doc_id] = _ensure_str(row.get("reference", ""))
    
    return result



def parse_reference_text(reference_text: str) -> Tuple[str, str]:
    """Parse a reference text (stored as a Python-dict-like string) to extract title and link.

    The OQA CSV stores the `reference` column as a string like:
      "{'authors': [...], 'doi': 'https://doi.org/...', 'meta': '...', 'title': 'Some Title'}"

    Returns (title, link) where link prefers `doi` if present, otherwise empty string.
    Fallbacks to simple regex-less extraction if parsing fails.
    """
    title: str = ""
    link: str = ""
    s = reference_text.strip()
    if not s:
        return title, link
    try:
        # Safely evaluate Python-literal-like dict
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            title = _ensure_str(obj.get("title", ""))
            # Prefer DOI if looks like a URL
            doi_val = _ensure_str(obj.get("doi", ""))
            if doi_val.startswith("http://") or doi_val.startswith("https://"):
                link = doi_val
            else:
                # some datasets may store other key for links
                link = doi_val
            # Optional: fallback to any other url-like fields if doi missing
            if not link:
                for k, v in obj.items():
                    vv = _ensure_str(v)
                    if vv.startswith("http://") or vv.startswith("https://"):
                        link = vv
                        break
            return title, link
    except Exception:
        # ignore and try fallback heuristics
        pass

    # Fallback: crude extraction for 'title': '...'
    try:
        # Extract between "title': '" and next "'"
        t_marker = "'title': '"
        ti = s.find(t_marker)
        if ti >= 0:
            ti += len(t_marker)
            te = s.find("'", ti)
            if te > ti:
                title = _ensure_str(s[ti:te])
        d_marker = "'doi': '"
        di = s.find(d_marker)
        if di >= 0:
            di += len(d_marker)
            de = s.find("'", di)
            if de > di:
                cand = _ensure_str(s[di:de])
                if cand:
                    link = cand
    except Exception:
        pass
    return title, link


def format_references_numbered(id_list: List[str], id_to_ref: Dict[str, str]) -> List[str]:
    """Format references as [N] TITLE LINK from a list of reference IDs and their raw text.

    Any missing fields are skipped gracefully. If no link, omit it.
    """
    output: List[str] = []
    for i, ref_id in enumerate(id_list, start=1):
        raw = id_to_ref.get(ref_id, "")
        title, link = parse_reference_text(raw)
        parts: List[str] = []
        if title:
            parts.append(title)
        if link:
            parts.append(link)
        if parts:
            output.append(f"[{i}] {' '.join(parts)}")
        else:
            # Fallback to raw if parsing failed
            if raw:
                output.append(f"[{i}] {raw}")
    return output

