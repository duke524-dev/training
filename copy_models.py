"""
Script to copy trained model files to Zeus directory.
Deletes old model files in destination before copying new ones.
"""

import os
import shutil
from glob import glob
from pathlib import Path
from datetime import datetime


# Configuration
SOURCE_DIR = "models"
DEST_DIR = "../Zeus/trained_models"


def get_latest_models(source_dir: str = "models") -> list[str]:
    """
    Get the latest model file for each variable.
    Returns list of file paths to the latest models.
    """
    pattern = os.path.join(source_dir, "model_*.pkl")
    all_files = glob(pattern)
    
    if not all_files:
        return []
    
    # Group by variable name (extract from filename)
    # Format: model_{variable}_{timestamp}.pkl
    models_by_variable = {}
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        # Extract variable name (everything between "model_" and last "_")
        parts = filename.replace(".pkl", "").split("_")
        if len(parts) >= 3:
            # Reconstruct variable name (handle variables with underscores)
            # Last two parts are date and time, everything before is variable
            variable_parts = parts[1:-2]  # Skip "model" and timestamp parts
            variable = "_".join(variable_parts)
            
            if variable not in models_by_variable:
                models_by_variable[variable] = []
            models_by_variable[variable].append(file_path)
    
    # Get latest file for each variable (by modification time)
    latest_models = []
    for variable, files in models_by_variable.items():
        latest = max(files, key=lambda f: os.path.getmtime(f))
        latest_models.append(latest)
    
    return sorted(latest_models)


def clean_destination(dest_dir: str):
    """
    Delete all .pkl files in the destination directory.
    """
    if not os.path.exists(dest_dir):
        return
    
    pattern = os.path.join(dest_dir, "*.pkl")
    old_files = glob(pattern)
    
    deleted_count = 0
    for file_path in old_files:
        try:
            os.remove(file_path)
            deleted_count += 1
            print(f"  Deleted: {os.path.basename(file_path)}")
        except OSError as e:
            print(f"  [WARNING] Could not delete {os.path.basename(file_path)}: {e}")
    
    return deleted_count


def copy_models(source_dir: str = "models", dest_dir: str = "../../Zeus/trained_models"):
    """
    Copy latest model files to destination directory.
    
    Args:
        source_dir: Source directory containing model files
        dest_dir: Destination directory in Zeus folder
    """
    print("=" * 70)
    print("Copying Models to Zeus Directory")
    print("=" * 70)
    
    # Check source directory
    if not os.path.exists(source_dir):
        print(f"\n[ERROR] Source directory not found: {source_dir}")
        return False
    
    # Get latest models
    print(f"\nScanning source directory: {source_dir}")
    model_files = get_latest_models(source_dir)
    
    if not model_files:
        print(f"  [ERROR] No model files found in {source_dir}")
        return False
    
    print(f"  Found {len(model_files)} model file(s)")
    for model_file in model_files:
        print(f"    - {os.path.basename(model_file)}")
    
    # Create destination directory if it doesn't exist
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    abs_dest_dir = os.path.abspath(dest_dir)
    print(f"\nDestination directory: {abs_dest_dir}")
    
    # Clean old files
    print("\nCleaning old model files from destination...")
    deleted_count = clean_destination(dest_dir)
    if deleted_count > 0:
        print(f"  Deleted {deleted_count} old file(s)")
    else:
        print("  No old files to delete")
    
    # Copy new models
    print("\nCopying new model files...")
    copied_count = 0
    failed_count = 0
    
    for model_file in model_files:
        filename = os.path.basename(model_file)
        dest_file = os.path.join(dest_dir, filename)
        
        try:
            shutil.copy2(model_file, dest_file)
            copied_count += 1
            print(f"  [OK] {filename}")
        except Exception as e:
            failed_count += 1
            print(f"  [ERROR] {filename}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Models found: {len(model_files)}")
    print(f"  Copied: {copied_count}")
    if failed_count > 0:
        print(f"  Failed: {failed_count}")
    
    # List copied files
    if copied_count > 0:
        print(f"\nCopied files to {abs_dest_dir}:")
        pattern = os.path.join(dest_dir, "*.pkl")
        copied_files = sorted(glob(pattern))
        for file_path in copied_files:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"  - {os.path.basename(file_path)} ({file_size:.2f} MB)")
    
    print("\n" + "=" * 70)
    if copied_count == len(model_files):
        print("SUCCESS: All models copied successfully!")
        print("=" * 70)
        return True
    else:
        print("PARTIAL SUCCESS: Some models failed to copy")
        print("=" * 70)
        return False


def verify_copy(source_dir: str = "models", dest_dir: str = "../../Zeus/trained_models"):
    """
    Verify that copied files match source files.
    """
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)
    
    source_files = get_latest_models(source_dir)
    dest_pattern = os.path.join(dest_dir, "*.pkl")
    dest_files = glob(dest_pattern)
    
    source_names = {os.path.basename(f) for f in source_files}
    dest_names = {os.path.basename(f) for f in dest_files}
    
    missing = source_names - dest_names
    extra = dest_names - source_names
    
    if not missing and not extra:
        print("  [OK] All files copied correctly")
        print(f"  Source: {len(source_names)} files")
        print(f"  Destination: {len(dest_names)} files")
        return True
    else:
        if missing:
            print(f"  [WARNING] Missing files in destination: {missing}")
        if extra:
            print(f"  [WARNING] Extra files in destination: {extra}")
        return False


def main():
    """Main function."""
    success = copy_models(SOURCE_DIR, DEST_DIR)
    
    if success:
        verify_copy(SOURCE_DIR, DEST_DIR)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

