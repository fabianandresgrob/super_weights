#!/usr/bin/env python3
"""
Script to download models from HuggingFace Hub based on model categories defined in model_list.json
"""
import json
import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
from tqdm import tqdm


def load_model_list(model_list_path):
    """Load the model list from JSON file."""
    try:
        with open(model_list_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Model list file not found at {model_list_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {model_list_path}")
        return None


def get_models_to_download(model_dict, categories):
    """Get list of models to download based on specified categories."""
    models_to_download = []
    
    for category in categories:
        if category in model_dict:
            models_to_download.extend(model_dict[category])
            print(f"Added {len(model_dict[category])} models from '{category}' category")
        else:
            print(f"Warning: Category '{category}' not found in model list")
    
    return models_to_download


def download_model(model_name, download_path, skip_existing=True):
    """Download a single model from HuggingFace Hub."""
    model_path = Path(download_path) / model_name.replace("/", "_")
    
    # Check if model already exists
    if skip_existing and model_path.exists():
        print(f"Model {model_name} already exists at {model_path}, skipping...")
        return True
    
    try:
        print(f"Downloading {model_name}...")
        snapshot_download(
            repo_id=model_name,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✓ Successfully downloaded {model_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download models from HuggingFace Hub based on categories in model_list.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default categories (regular_models and moe_models)
  python download_models.py
  
  # Download only regular models
  python download_models.py --categories regular_models
  
  # Download multiple categories
  python download_models.py --categories regular_models moe_models large_moe_models
  
  # Download to custom path
  python download_models.py --download-path /path/to/models --categories regular_models
        """
    )
    
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["regular_models", "moe_models"],
        help="Categories of models to download (default: regular_models moe_models)"
    )
    
    parser.add_argument(
        "--download-path",
        type=str,
        default="../models/",
        help="Path where to download the models (default: ../models/)"
    )
    
    parser.add_argument(
        "--model-list",
        type=str,
        default="utils/model_list.json",
        help="Path to the model list JSON file (default: utils/model_list.json)"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip downloading models that already exist (default: True)"
    )
    
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force re-download even if model already exists"
    )
    
    args = parser.parse_args()
    
    # Handle force redownload
    if args.force_redownload:
        args.skip_existing = False
    
    # Load model list
    model_dict = load_model_list(args.model_list)
    if model_dict is None:
        return 1
    
    # Get models to download
    models_to_download = get_models_to_download(model_dict, args.categories)
    
    if not models_to_download:
        print("No models to download.")
        return 0
    
    # Create download directory if it doesn't exist
    download_path = Path(args.download_path)
    download_path.mkdir(parents=True, exist_ok=True)
    print(f"Download directory: {download_path.absolute()}")
    
    # Print summary
    print(f"\nPlan to download {len(models_to_download)} models:")
    for i, model in enumerate(models_to_download, 1):
        print(f"  {i}. {model}")
    
    # Confirm download
    response = input(f"\nProceed with download? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Download cancelled.")
        return 0
    
    # Download models
    print(f"\nStarting download of {len(models_to_download)} models...")
    successful_downloads = 0
    failed_downloads = 0
    
    for i, model in enumerate(models_to_download, 1):
        print(f"\n[{i}/{len(models_to_download)}] Processing {model}")
        if download_model(model, download_path, args.skip_existing):
            successful_downloads += 1
        else:
            failed_downloads += 1
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Download Summary:")
    print(f"  Successful: {successful_downloads}")
    print(f"  Failed: {failed_downloads}")
    print(f"  Total: {len(models_to_download)}")
    
    if failed_downloads > 0:
        print(f"\nSome downloads failed. You may want to retry with --force-redownload flag.")
        return 1
    
    print(f"\nAll models downloaded successfully to {download_path.absolute()}")
    return 0


if __name__ == "__main__":
    exit(main())
