"""
Load CSV files into Qdrant collections with hybrid search support.

This script loads medical knowledge base CSV files into separate Qdrant collections,
creating embeddings using dense, sparse (BM25), and late interaction (ColBERT) models.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

from utils.qdrant.config import (
    QDRANT_URL,
    FASTEMBED_CACHE,
    COLLECTION_BNDTD, COLLECTION_BSNT, COLLECTION_BNRHM, COLLECTION_BSRHM,
    VECTOR_DENSE, VECTOR_LATE,
    get_vector_params, get_sparse_vector_params
)
from utils.qdrant.embeddings import EmbeddingService
from utils.qdrant.client import get_client

# Constants
BATCH_SIZE = 64
CSV_BASE_PATH = "medical_knowledge_base"

# Collection configurations: collection_name -> (csv_filename, required_columns)
COLLECTION_CONFIGS = {
    COLLECTION_BNDTD: ("bndtd.csv", ["DEMUC", "CHUDECON", "CAUHOI", "CAUTRALOI", "GIAITHICH"]),
    COLLECTION_BSNT: ("bsnt.csv", ["DEMUC", "CHUDECON", "CAUHOI", "CAUTRALOI", "GIAITHICH"]),
    COLLECTION_BNRHM: ("bnrhm.csv", ["DEMUC", "CHUDECON", "CAUHOI", "CAUTRALOI", "GIAITHICH"]),
    COLLECTION_BSRHM: ("bsrhm.csv", ["DEMUC", "CHUDECON", "CAUHOI", "CAUTRALOI"]),  # No GIAITHICH
}

# Abbreviation expansion mapping
ABBREVIATION_MAP = {
    "ĐTĐ": "Đái Tháo Đường",
    "DTĐ": "Đái Tháo Đường",
    "tiểu đường": "Đái Tháo Đường",
     "Tiểu đường": "Đái Tháo Đường"
}


def expand_abbreviations(text: str) -> str:
    """
    Expand common medical abbreviations in text.

    Args:
        text: Input text that may contain abbreviations

    Returns:
        Text with abbreviations expanded
    """
    if not text:
        return text

    expanded_text = text
    for abbr, full_form in ABBREVIATION_MAP.items():
        expanded_text = expanded_text.replace(abbr, full_form)
    return expanded_text


class EmbeddingModels:
    """
    Legacy Adapter: Container for embedding models with lazy loading and cache support.
    Uses EmbeddingService under the hood.
    """

    def __init__(self):
        self.dense_model = None
        self.sparse_model = None
        self.late_interaction_model = None

    def load(self):
        """Load all embedding models from cache directory."""
        print(f"Loading embedding models from cache: {FASTEMBED_CACHE}")
        self.dense_model, self.sparse_model, self.late_interaction_model = EmbeddingService.get_models()
        print("All models loaded from cache successfully.\n")


def load_csv_data(csv_path: str, required_columns: List[str]) -> List[Dict[str, str]]:
    """
    Load CSV data and extract required columns.
    """
    print(f"Loading CSV file: {csv_path}")

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  - Loaded {len(df)} rows")

    # Check if all required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")

    # Select required columns and fill NaN values
    df_filtered = df[required_columns].fillna("")

    # Convert to list of dictionaries
    docs = df_filtered.to_dict(orient='records')
    print(f"  - Extracted {len(docs)} documents\n")

    return docs


def generate_embeddings(
    docs: List[Dict[str, str]],
    models: EmbeddingModels
) -> Tuple[List, List, List]:
    """
    Generate embeddings for documents using all three models.
    """
    print("Generating embeddings...")

    # Extract questions for embedding
    questions = [doc["CAUHOI"] for doc in docs]

    # Use the models attached to the wrapper
    print("  - Generating dense embeddings...")
    dense_embeddings = list(models.dense_model.embed(questions))

    print("  - Generating sparse embeddings (BM25)...")
    sparse_embeddings = list(models.sparse_model.embed(questions))

    print("  - Generating late interaction embeddings (ColBERT)...")
    late_interaction_embeddings = list(models.late_interaction_model.embed(questions))

    print(f"  - Generated embeddings for {len(questions)} documents\n")

    return dense_embeddings, sparse_embeddings, late_interaction_embeddings


def collection_has_data(client: QdrantClient, collection_name: str) -> Tuple[bool, int]:
    """
    Check if a collection exists and has data.
    """
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(col.name == collection_name for col in collections)

        if not exists:
            return False, 0

        # Check if collection has data
        collection_info = client.get_collection(collection_name)
        points_count = collection_info.points_count

        return points_count > 0, points_count

    except Exception:
        return False, 0


def create_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int,
    late_dim: int,
    recreate: bool = False
) -> bool:
    """
    Create a Qdrant collection with hybrid search configuration.
    """
    print(f"Setting up collection: {collection_name}")

    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(col.name == collection_name for col in collections)

    if exists:
        if recreate:
            print(f"  - Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            print(f"  - Collection already exists, skipping creation\n")
            return False

    print(f"  - Creating collection with hybrid search configuration...")

    # Use centralized config helpers
    # Note: we are passing qdrant_client.models to the helpers.
    # But wait, config.py doesn't import models to avoid circular dep (or so I wrote).
    # Actually config.py imports nothing from qdrant_client except type hints if I'm careful.
    # Let's check config.py content again.

    client.create_collection(
        collection_name=collection_name,
        vectors_config=get_vector_params(models),
        sparse_vectors_config=get_sparse_vector_params(models)
    )

    print(f"  - Collection created successfully\n")
    return True


def prepare_points(
    docs: List[Dict[str, str]],
    dense_embeddings: List,
    sparse_embeddings: List,
    late_interaction_embeddings: List
) -> List[PointStruct]:
    """
    Prepare PointStruct objects for uploading to Qdrant.
    """
    print("Preparing points for upload...")

    points = []
    for idx, (dense_emb, sparse_emb, late_emb, doc) in enumerate(
        zip(dense_embeddings, sparse_embeddings, late_interaction_embeddings, docs)
    ):
        # Expand abbreviations in CAUHOI field for better searchability
        cauhoi = doc.get("CAUHOI", "")
        cauhoi_expanded = expand_abbreviations(cauhoi)

        # Prepare payload with all available fields
        payload = {
            "DEMUC": doc.get("DEMUC", ""),
            "CHUDECON": doc.get("CHUDECON", ""),
            "CAUHOI": cauhoi_expanded,
            "CAUTRALOI": doc.get("CAUTRALOI", ""),
            "GIAITHICH": doc.get("GIAITHICH", "")  # Will be empty string if not present
        }

        point = PointStruct(
            id=idx,
            vector={
                VECTOR_DENSE: dense_emb,
                "bm25": sparse_emb.as_object(), # Using hardcoded string "bm25" in original, which matches VECTOR_SPARSE?
                # In config.py: VECTOR_SPARSE = "bm25". Correct.
                VECTOR_LATE: late_emb,
            },
            payload=payload
        )
        points.append(point)

    print(f"  - Prepared {len(points)} points\n")
    return points


def upsert_in_batches(
    client: QdrantClient,
    collection_name: str,
    points: List[PointStruct],
    batch_size: int = BATCH_SIZE
) -> None:
    """
    Upload points to Qdrant collection in batches.
    """
    print(f"Uploading {len(points)} points in batches of {batch_size}...")

    total_batches = (len(points) + batch_size - 1) // batch_size

    for i in range(0, len(points), batch_size):
        batch_num = i // batch_size + 1
        chunk = points[i:i + batch_size]

        client.upsert(
            collection_name=collection_name,
            points=chunk,
        )

        print(f"  - Uploaded batch {batch_num}/{total_batches} ({len(chunk)} points)")

    print(f"  - Upload complete!\n")


def load_single_collection(
    collection_name: str,
    csv_filename: str,
    required_columns: List[str],
    client: QdrantClient,
    models: EmbeddingModels,
    recreate: bool = False
) -> bool:
    """
    Load a single collection from CSV into Qdrant.
    """
    print("=" * 70)
    print(f"LOADING COLLECTION: {collection_name}")
    print("=" * 70)

    try:
        # Check if collection already has data
        has_data, points_count = collection_has_data(client, collection_name)

        if has_data and not recreate:
            print(f"  - Collection '{collection_name}' already has {points_count} points")
            print(f"  - Skipping load (use --recreate to force reload)\n")
            return True

        # Build CSV path
        csv_path = Path(CSV_BASE_PATH) / csv_filename

        # Load CSV data
        docs = load_csv_data(str(csv_path), required_columns)

        # Generate embeddings
        dense_embs, sparse_embs, late_embs = generate_embeddings(docs, models)

        # Create collection
        dense_dim = len(dense_embs[0])
        late_dim = len(late_embs[0][0])  # Multi-vector, get first vector dimension

        created = create_collection(client, collection_name, dense_dim, late_dim, recreate)

        # Prepare points
        points = prepare_points(docs, dense_embs, sparse_embs, late_embs)

        # Upload in batches
        upsert_in_batches(client, collection_name, points)

        print(f" Successfully loaded collection: {collection_name}")
        print(f"  - Total documents: {len(docs)}")
        print(f"  - CSV source: {csv_filename}\n")

        return True

    except Exception as e:
        print(f" Error loading collection {collection_name}: {e}\n")
        return False


def load_all_collections(
    client: QdrantClient,
    models: EmbeddingModels,
    collections: List[str] = None,
    recreate: bool = False
) -> Dict[str, bool]:
    """
    Load all or specified collections into Qdrant.
    """
    results = {}

    # Determine which collections to load
    collections_to_load = collections if collections else list(COLLECTION_CONFIGS.keys())

    print(f"\nStarting to load {len(collections_to_load)} collection(s)...\n")

    for collection_name in collections_to_load:
        if collection_name not in COLLECTION_CONFIGS:
            print(f" Unknown collection: {collection_name}, skipping...\n")
            results[collection_name] = False
            continue

        csv_filename, required_columns = COLLECTION_CONFIGS[collection_name]

        success = load_single_collection(
            collection_name,
            csv_filename,
            required_columns,
            client,
            models,
            recreate
        )

        results[collection_name] = success

    return results


def print_summary(results: Dict[str, bool]) -> None:
    """Print summary of loading results."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]

    print(f"Total collections processed: {len(results)}")
    print(f"Successful/Skipped: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print(f"\nSuccessfully loaded or skipped (already had data):")
        for name in successful:
            print(f"  - {name}")

    if failed:
        print(f"\nFailed to load:")
        for name in failed:
            print(f"  - {name}")

    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load CSV files into Qdrant collections with hybrid search support"
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        choices=list(COLLECTION_CONFIGS.keys()),
        help="Specific collections to load (default: all)"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collections if they already exist"
    )
    parser.add_argument(
        "--url",
        default=QDRANT_URL,
        help=f"Qdrant server URL (default: {QDRANT_URL})"
    )

    args = parser.parse_args()

    try:
        # Initialize Qdrant client
        # In main, we can ignore the passed URL if we want strict singleton usage,
        # but for CLI flexibility we might want to allow it.
        # However, our ClientManager reads from env.
        # If args.url is different, we might have an issue.
        # For consistency with the refactor, let's use the singleton client.
        print(f"Connecting to Qdrant (URL from ENV: {QDRANT_URL})...")
        if args.url != QDRANT_URL:
             print(f"Warning: --url argument ignored in favor of centralized configuration.")

        client = get_client()
        print("Connected successfully.\n")

        # Load embedding models
        models = EmbeddingModels()
        models.load()

        # Load collections
        results = load_all_collections(
            client,
            models,
            collections=args.collections,
            recreate=args.recreate
        )

        # Print summary
        print_summary(results)

        # Exit with appropriate code
        if all(results.values()):
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
