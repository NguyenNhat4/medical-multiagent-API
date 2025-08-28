import os
import random
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode


KB_COLUMNS = [
    "ĐỀ MỤC",
    "CHỦ  ĐỀ  CON",
    "MÃ SỐ",
    "CÂU HỎI",
    "CÂU  TRẢ    LỜI",
    "keywords",
]

ROLE_TO_CSV = {
    "bệnh nhân đái tháo đường": "bndtd.csv",
    "bác sĩ nội tiết": "bsnt.csv", 
    "bệnh nhân răng hàm mặt": "bnrhm.csv",
    "bác sĩ răng hàm mặt": "bsrhm.csv",
    "bệnh nhân nha khoa": "bnrhm.csv",
    "bác sĩ nha khoa": "bsrhm.csv",    
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


class KnowledgeBaseIndex:
    def __init__(self, kb_dir: str = "medical_knowledge_base") -> None:
        self.kb_dir = kb_dir
        self.df: pd.DataFrame = pd.DataFrame()
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        # Store individual CSV dataframes for role-based access
        self.role_dataframes: Dict[str, pd.DataFrame] = {}
        self._load()

    def _load(self) -> None:
        frames: List[pd.DataFrame] = []
        if not os.path.isdir(self.kb_dir):
            raise FileNotFoundError(f"Knowledge base directory not found: {self.kb_dir}")

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
            
            frames.append(df)

        if not frames:
            raise ValueError("No CSV files loaded from knowledge base directory")

        merged = pd.concat(frames, ignore_index=True)

        # Combined field for retrieval
        merged["combined"] = (
            merged["ĐỀ MỤC"]
            + " \n "
            + merged["CHỦ  ĐỀ  CON"]
            + " \n "
            + merged["CÂU HỎI"]
            + " \n "
            + merged["CÂU  TRẢ    LỜI"]
            + " \n "
            + merged["keywords"]
        )

        merged["combined_norm"] = merged["combined"].apply(_normalize_accents)

        self.df = merged

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        self.matrix = self.vectorizer.fit_transform(self.df["combined_norm"])  # type: ignore[arg-type]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        assert self.vectorizer is not None and self.matrix is not None
        q = _normalize_accents(_normalize_text(query))
        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        idx = sims.argsort()[::-1][:top_k]
        results: List[Dict[str, Any]] = []
        for i in idx:
            row = self.df.iloc[int(i)]
            results.append(
                {
                    "score": float(sims[int(i)]),
                    "de_muc": row.get("ĐỀ MỤC", ""),
                    "chu_de_con": row.get("CHỦ  ĐỀ  CON", ""),
                    "ma_so": row.get("MÃ SỐ", ""),
                    "cau_hoi": row.get("CÂU HỎI", ""),
                    "cau_tra_loi": row.get("CÂU  TRẢ    LỜI", ""),
                    "keywords": row.get("keywords", ""),
                }
            )
        return results

    def best_score(self, query: str) -> float:
        hits = self.search(query, top_k=1)
        return hits[0]["score"] if hits else 0.0

    def get_random_by_role(self, role: str, amount: int = 5) -> List[Dict[str, Any]]:
        """Get random entries from CSV file based on user role"""
        
        # Find the appropriate CSV file for this role
        csv_file = None
        role_lower = role.lower()
        
        for role_key, file_name in ROLE_TO_CSV.items():
            if role_key in role_lower:
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
                "de_muc": row.get("ĐỀ MỤC", ""),
                "chu_de_con": row.get("CHỦ  ĐỀ  CON", ""),
                "ma_so": row.get("MÃ SỐ", ""),
                "cau_hoi": row.get("CÂU HỎI", ""),
                "cau_tra_loi": row.get("CÂU  TRẢ    LỜI", ""),
                "keywords": row.get("keywords", ""),
            })
        
        return results


_KB_INDEX: KnowledgeBaseIndex | None = None

def get_kb() -> KnowledgeBaseIndex:
    global _KB_INDEX
    if _KB_INDEX is None:
        _KB_INDEX = KnowledgeBaseIndex()
    return _KB_INDEX


def retrieve(query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], float]:
    kb = get_kb()
    results = kb.search(query, top_k=top_k)
    score = results[0]["score"] if results else 0.0
    return results, score


def retrieve_random_by_role(role: str, amount: int = 5) -> List[Dict[str, Any]]:
    """Retrieve random entries from KB based on user role"""
    kb = get_kb()
    return kb.get_random_by_role(role, amount)


