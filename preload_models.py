from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
import sys
import os
import shutil
import time
import glob

def clear_corrupted_model(cache_dir, model_identifier):
    """Clear all potential corrupted model files and cache."""
    patterns = [
        f'models--*{model_identifier}*',
        f'*{model_identifier}*',
    ]

    for pattern in patterns:
        for path in glob.glob(os.path.join(cache_dir, pattern)):
            try:
                if os.path.isdir(path):
                    print(f'🗑️  Removing corrupted directory: {path}')
                    shutil.rmtree(path)
                else:
                    print(f'🗑️  Removing corrupted file: {path}')
                    os.remove(path)
            except Exception as e:
                print(f'⚠️  Could not remove {path}: {e}')

def download_model_with_retry(model_class, model_name, cache_dir, max_retries=5, initial_delay=2):
    """
    Download and verify a model with exponential backoff retry logic.

    Args:
        model_class: FastEmbed model class (TextEmbedding, SparseTextEmbedding, etc.)
        model_name: Name of the model to download
        cache_dir: Cache directory path
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds between retries (doubles each retry)

    Returns:
        Loaded model instance
    """
    delay = initial_delay

    for attempt in range(1, max_retries + 1):
        try:
            print(f'📥 Attempt {attempt}/{max_retries}: Downloading {model_name}...')

            # Load model with CPU provider for stability
            if model_class == TextEmbedding:
                model = model_class(
                    model_name,
                    cache_dir=cache_dir,
                    providers=['CPUExecutionProvider']
                )
            else:
                model = model_class(model_name, cache_dir=cache_dir)

            # Verify the model works by running a test embedding
            print(f'🔍 Verifying {model_name}...')
            test_result = list(model.embed(["test embedding verification"]))

            if not test_result:
                raise ValueError("Model returned empty embeddings during verification")

            print(f'✅ {model_name} downloaded and verified successfully!')
            return model

        except Exception as e:
            error_msg = str(e)
            print(f'❌ Attempt {attempt} failed for {model_name}: {error_msg}')

            # Check if it's a corruption error
            is_corruption = any(keyword in error_msg.lower() for keyword in [
                'corrupted',
                'modelproto does not have a graph',
                'onnxruntimeerror',
                'download',
                'incomplete'
            ])

            if is_corruption or 'could not download' in error_msg.lower():
                print(f'🧹 Detected corruption/download issue, clearing cache...')
                # Extract model identifier for cleanup
                model_id = model_name.split('/')[-1] if '/' in model_name else model_name
                clear_corrupted_model(cache_dir, model_id)

            if attempt < max_retries:
                print(f'⏳ Waiting {delay} seconds before retry...')
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f'💥 Failed to download {model_name} after {max_retries} attempts')
                raise

def download_models():
    cache_dir = os.getenv('FASTEMBED_CACHE_PATH', '/app/models')

    try:
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        print(f'\n📁 Cache directory: {cache_dir}')
        print(f'🌐 Starting model downloads with retry logic...\n')

        # Download models with retry logic
        models = [
            (TextEmbedding, 'sentence-transformers/all-MiniLM-L6-v2', 'Dense model'),
            (SparseTextEmbedding, 'Qdrant/bm25', 'Sparse model (BM25)'),
            (LateInteractionTextEmbedding, 'colbert-ir/colbertv2.0', 'Late interaction model (ColBERTv2)')
        ]

        for model_class, model_name, description in models:
            print(f'\n{"="*60}')
            print(f'📦 {description}')
            print(f'{"="*60}')
            download_model_with_retry(model_class, model_name, cache_dir)

        print(f'\n{"="*60}')
        print('✅ All models cached and verified successfully!')
        print(f'{"="*60}\n')

        # Print cache directory contents for debugging
        print(f'📊 Cache directory contents:')
        try:
            for item in sorted(os.listdir(cache_dir)):
                item_path = os.path.join(cache_dir, item)
                if os.path.isdir(item_path):
                    # Count files in directory
                    file_count = sum(1 for _ in glob.glob(f'{item_path}/**/*', recursive=True))
                    print(f'  📂 {item}/ ({file_count} files)')
                else:
                    size = os.path.getsize(item_path)
                    size_mb = size / (1024 * 1024)
                    print(f'  📄 {item} ({size_mb:.2f} MB)')
        except Exception as e:
            print(f'  ⚠️  Could not list directory contents: {e}')

        print(f'\n🎉 Model preloading completed successfully!\n')

    except Exception as e:
        print(f'\n💥 FATAL ERROR: Model download failed')
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

        # Clean up cache directory on fatal error
        print(f'\n🧹 Cleaning up cache directory due to fatal error...')
        try:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir)
                print(f'✅ Cache directory cleaned')
        except Exception as cleanup_error:
            print(f'⚠️  Could not clean cache: {cleanup_error}')

        sys.exit(1)

if __name__ == "__main__":
    download_models()