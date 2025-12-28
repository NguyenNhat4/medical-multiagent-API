#!/usr/bin/env python3
"""
Script to clear corrupted embedding model cache.

Use this script if you encounter model corruption errors and need to clear the cache
without rebuilding the entire Docker image.

Usage:
    # Clear all model caches
    python clear_model_cache.py

    # Clear specific model
    python clear_model_cache.py --model all-MiniLM-L6-v2
"""

import os
import shutil
import sys
import argparse
from pathlib import Path


def get_cache_dir():
    """Get the model cache directory."""
    return os.getenv('FASTEMBED_CACHE_PATH', './models')


def list_cached_models(cache_dir):
    """List all cached models."""
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return []

    models = []
    for item in os.listdir(cache_dir):
        item_path = os.path.join(cache_dir, item)
        if os.path.isdir(item_path) and item.startswith('models--'):
            size = get_dir_size(item_path)
            size_mb = size / (1024 * 1024)
            models.append((item, size_mb))

    return models


def get_dir_size(path):
    """Calculate total size of directory."""
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_dir_size(entry.path)
    return total


def clear_model_cache(cache_dir, model_name=None):
    """
    Clear model cache.

    Args:
        cache_dir: Path to cache directory
        model_name: Specific model to clear (None = clear all)
    """
    if not os.path.exists(cache_dir):
        print(f"✅ Cache directory does not exist: {cache_dir}")
        return

    if model_name:
        # Clear specific model
        patterns = [
            f"models--*{model_name}*",
            f"*{model_name}*",
        ]

        cleared = False
        for pattern in patterns:
            for item in os.listdir(cache_dir):
                if pattern.replace('*', '') in item:
                    item_path = os.path.join(cache_dir, item)
                    if os.path.isdir(item_path):
                        print(f"🗑️  Removing: {item}")
                        try:
                            shutil.rmtree(item_path)
                            cleared = True
                        except Exception as e:
                            print(f"❌ Error removing {item}: {e}")

        if cleared:
            print(f"✅ Cleared cache for model: {model_name}")
        else:
            print(f"⚠️  No cache found for model: {model_name}")

    else:
        # Clear all models
        print(f"🗑️  Clearing entire cache directory: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            print("✅ Cache cleared successfully")
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Clear embedding model cache to fix corruption issues"
    )
    parser.add_argument(
        '--model',
        help='Specific model to clear (e.g., all-MiniLM-L6-v2, bm25, colbertv2.0). Omit to clear all.',
        default=None
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List cached models without clearing'
    )
    parser.add_argument(
        '--cache-dir',
        help='Override cache directory path',
        default=None
    )

    args = parser.parse_args()

    # Determine cache directory
    cache_dir = args.cache_dir if args.cache_dir else get_cache_dir()

    print("=" * 70)
    print("Embedding Model Cache Manager")
    print("=" * 70)
    print(f"Cache directory: {cache_dir}\n")

    if args.list:
        # List models
        models = list_cached_models(cache_dir)
        if models:
            print(f"Found {len(models)} cached model(s):\n")
            for model_name, size_mb in models:
                print(f"  📦 {model_name} ({size_mb:.2f} MB)")
        else:
            print("No cached models found.")
    else:
        # Clear cache
        if args.model:
            print(f"Clearing cache for model: {args.model}\n")
        else:
            print("⚠️  WARNING: This will clear ALL cached models!\n")
            response = input("Continue? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)

        clear_model_cache(cache_dir, args.model)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
