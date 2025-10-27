import os
import random
import re
from typing import List, Dict, Any, Tuple, Optional
from functools import lru_cache
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from unidecode import unidecode
from .role_enum import RoleEnum

KB_COLUMNS = [
    "DEMUC",
    "CHUDECON",
    "MASO",
    "CAUHOI",
    "CAUTRALOI",
    "keywords",
    "GIAITHICH",  # Optional column - not all CSV files have this
]

ROLE_TO_CSV = {
    RoleEnum.PATIENT_DIABETES.value: "bndtd.csv",
    RoleEnum.DOCTOR_ENDOCRINE.value: "bsnt.csv", 
    RoleEnum.PATIENT_DENTAL.value: "bnrhm.csv",
    RoleEnum.DOCTOR_DENTAL.value: "bsrhm.csv",
}


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\n", " ").replace("\r", " ")
    s = " ".join(s.split())
    return s


def _normalize_accents(text: str) -> str:
    return unidecode(text.lower())


def _collapse_spaces(text: str) -> str:
    """Collapse multiple spaces into single spaces for robust column matching."""
    return " ".join(str(text).strip().split())


def _tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 by normalizing Vietnamese and removing special chars."""
    s = unidecode(str(text)).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [t for t in s.split() if t]


class KnowledgeBaseIndex:
    def __init__(self, kb_dir: str = "medical_knowledge_base") -> None:
        self.kb_dir = kb_dir
        self.df: pd.DataFrame = pd.DataFrame()
        self.bm25: BM25Okapi | None = None
        # Store individual CSV dataframes for role-based access
        self.role_dataframes: Dict[str, pd.DataFrame] = {}
        # Store role-specific BM25 indices
        self.role_bm25s: Dict[str, BM25Okapi] = {}
        self._load()

    def _load(self) -> None:
        frames: List[pd.DataFrame] = []
        if not os.path.isdir(self.kb_dir):
            raise FileNotFoundError(f"Knowledge base directory not found: {self.kb_dir}")

        # Create reverse mapping from CSV filename to role enum value
        csv_to_role = {filename: role_key for role_key, filename in ROLE_TO_CSV.items()}

        for name in os.listdir(self.kb_dir):
            if not name.lower().endswith(".csv"):
                continue
            path = os.path.join(self.kb_dir, name)
            df = None
            # Try utf-8 first, then fallback
            for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
                try:
                    df = pd.read_csv(path, encoding=enc)
                    break
                except Exception:
                    continue
            if df is None:
                continue

            # Standardize column names by stripping and collapsing spaces
            colmap: Dict[str, str] = {}
            for c in df.columns:
                key = _collapse_spaces(c)
                colmap[c] = key
            df = df.rename(columns=colmap)

            # Handle STT → MASO mapping for bsrhm.csv compatibility
            if "STT" in df.columns and "MASO" not in df.columns:
                df["MASO"] = df["STT"]

            # Ensure required columns exist under BOTH original (possibly multi-space)
            # and single-space-collapsed variants for downstream compatibility.
            ensured_cols: Dict[str, str] = {}
            for original_col in KB_COLUMNS:
                collapsed_col = _collapse_spaces(original_col)

                # If collapsed exists, prefer its data
                if collapsed_col in df.columns:
                    # Mirror into original_col name if missing
                    if original_col not in df.columns:
                        df[original_col] = df[collapsed_col]
                    ensured_cols[original_col] = original_col
                    continue

                # If original exists (rare at this point), mirror to collapsed
                if original_col in df.columns:
                    df[collapsed_col] = df[original_col]
                    ensured_cols[original_col] = original_col
                    continue

                # Neither exists – create empty columns for both names
                df[collapsed_col] = ""
                df[original_col] = ""
                ensured_cols[original_col] = original_col

            # Keep only known columns
            df = df[KB_COLUMNS]
            
            # Clean and prepare fields
            for col in KB_COLUMNS:
                df[col] = df[col].apply(_normalize_text)
            
            # Store individual CSV dataframe for role-based access
            self.role_dataframes[name] = df.copy()

            # Create role-specific BM25 index if this CSV maps to a role
            if name in csv_to_role and len(df) > 0:
                role_key = csv_to_role[name]

                # Create combined field for this role's data
                role_df = df.copy()
                role_df["combined"] = (
                    role_df["DEMUC"]
                    + " \n "
                    + role_df["CHUDECON"]
                    + " \n "
                    + role_df["CAUHOI"]
                    + " \n "
                    + role_df["CAUTRALOI"]
                    + " \n "
                    + role_df["keywords"]
                    + " \n "
                    + role_df["GIAITHICH"]  # Include explanation if available
                )
                role_df["combined_norm"] = role_df["combined"].apply(_normalize_accents)

                # Tokenize corpus for BM25
                tokenized_corpus = [_tokenize(doc) for doc in role_df["combined_norm"]]

                # Create BM25 index for this role
                self.role_bm25s[role_key] = BM25Okapi(tokenized_corpus)
            
            frames.append(df)

        if not frames:
            raise ValueError("No CSV files loaded from knowledge base directory")

        # Create merged dataframe for fallback search
        merged = pd.concat(frames, ignore_index=True)

        # Combined field for retrieval
        merged["combined"] = (
            merged["DEMUC"]
            + " \n "
            + merged["CHUDECON"]
            + " \n "
            + merged["CAUHOI"]
            + " \n "
            + merged["CAUTRALOI"]
            + " \n "
            + merged["keywords"]
            + " \n "
            + merged["GIAITHICH"]  # Include explanation if available
        )

        merged["combined_norm"] = merged["combined"].apply(_normalize_accents)

        self.df = merged

        # Create general BM25 index for fallback search
        tokenized_corpus_general = [_tokenize(doc) for doc in self.df["combined_norm"]]
        self.bm25 = BM25Okapi(tokenized_corpus_general)

    def search(self, query: str, role: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        # Role-specific search
        if role and role in self.role_bm25s:
            bm25_index = self.role_bm25s[role]
            # Find corresponding dataframe
            csv_file = ROLE_TO_CSV.get(role)
            if csv_file and csv_file in self.role_dataframes:
                source_df = self.role_dataframes[csv_file]
            else:
                # Fallback to general search if no role-specific data
                assert self.bm25 is not None
                bm25_index = self.bm25
                source_df = self.df
        else:
            # General search across all data
            assert self.bm25 is not None
            bm25_index = self.bm25
            source_df = self.df

        # Tokenize query
        q = _normalize_accents(_normalize_text(query))
        q_tokens = _tokenize(q)

        # Get BM25 scores
        scores = bm25_index.get_scores(q_tokens)
        scores = np.array(scores, dtype=np.float32)

        if scores.size == 0:
            return []

        k = int(min(top_k, scores.shape[0]))
        # Use argpartition for O(n) top-k selection, then sort those k
        idx_part = np.argpartition(scores, -k)[-k:]
        idx = idx_part[np.argsort(scores[idx_part])[::-1]]

        results: List[Dict[str, Any]] = []
        for i in idx:
            row = source_df.iloc[int(i)]
            results.append(
                {
                    "score": float(scores[int(i)]),
                    "de_muc": row.get("DEMUC", ""),
                    "chu_de_con": row.get("CHUDECON", ""),
                    "ma_so": row.get("MASO", ""),
                    "cau_hoi": row.get("CAUHOI", ""),
                    "cau_tra_loi": row.get("CAUTRALOI", ""),
                    "keywords": row.get("keywords", ""),
                    "giai_thich": row.get("GIAITHICH", ""),  # Add explanation field
                }
            )
        return results

    def best_score(self, query: str, role: Optional[str] = None) -> float:
        hits = self.search(query, role=role, top_k=1)
        return hits[0]["score"] if hits else 0.0

    def get_random_by_role(self, role: str, amount: int = 5) -> List[Dict[str, Any]]:
        """Get random entries from CSV file based on user role"""
        
        # Find the appropriate CSV file for this role
        csv_file = None
        role_lower = role.lower()
        
        for role_key, file_name in ROLE_TO_CSV.items():
            if role_key == role_lower:
                csv_file = file_name
                break
        
        if not csv_file or csv_file not in self.role_dataframes:
            # Fallback to random from all data if no specific file found
            if len(self.df) == 0:
                return []
            sample_size = min(amount, len(self.df))
            sampled_df = self.df.sample(n=sample_size)
        else:
            # Get random entries from role-specific CSV
            role_df = self.role_dataframes[csv_file]
            if len(role_df) == 0:
                return []
            sample_size = min(amount, len(role_df))
            sampled_df = role_df.sample(n=sample_size)
        
        results: List[Dict[str, Any]] = []
        for _, row in sampled_df.iterrows():
            results.append({
                "score": 1.0,  # Random selection, so full score
                "de_muc": row.get("DEMUC", ""),
                "chu_de_con": row.get("CHUDECON", ""),
                "ma_so": row.get("MASO", ""),
                "cau_hoi": row.get("CAUHOI", ""),
                "cau_tra_loi": row.get("CAUTRALOI", ""),
                "keywords": row.get("keywords", ""),
                "giai_thich": row.get("GIAITHICH", ""),  # Add explanation field
            })

        return results


_KB_INDEX: KnowledgeBaseIndex | None = None

def get_kb() -> KnowledgeBaseIndex:
    global _KB_INDEX
    if _KB_INDEX is None:
        _KB_INDEX = KnowledgeBaseIndex()
    return _KB_INDEX


@lru_cache(maxsize=4096)
def _cached_search(query: str, role: Optional[str], top_k: int) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
    """Cacheable wrapper for KB search returning a hashable structure."""
    kb = get_kb()
    results = kb.search(query, role=role, top_k=top_k)
    # Convert list[dict] to tuple of sorted tuples for hashing
    return tuple(tuple(sorted(item.items())) for item in results)


def retrieve(query: str, role: Optional[str] = None, top_k: int = 5) -> Tuple[List[Dict[str, Any]], float]:
    # Use cached results to avoid recomputation for identical queries
    cached = _cached_search(query, role, top_k)
    results: List[Dict[str, Any]] = [dict(items) for items in cached]
    score = results[0]["score"] if results else 0.0
    return results, score


def retrieve_random_by_role(role: str, amount: int = 5) -> List[Dict[str, Any]]:
    """Retrieve random entries from KB based on user role"""
    kb = get_kb()
    return kb.get_random_by_role(role, amount)


