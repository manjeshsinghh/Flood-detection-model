"""
Script to organize River Basin images into flood prone and non-flood prone folders.
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def organize_images():
    """Organize all images in River Basin folder into flood prone and non-flood prone subfolders."""
    
    base_dir = Path("River Basin")
    
    # Create target directories
    flood_prone_dir = base_dir / "flood prone"
    non_flood_prone_dir = base_dir / "non-flood prone"
    
    flood_prone_dir.mkdir(exist_ok=True)
    non_flood_prone_dir.mkdir(exist_ok=True)
    
    # Define source directories for flood images
    flood_sources = [
        base_dir / "Flood_Images",
        base_dir / "archive (1)" / "flood",
        base_dir / "barren_land_flood",
        base_dir / "erosion_flood",
        base_dir / "riverbank_agri_flood",
        base_dir / "waterpollution_flood"
    ]
    
    # Define source directory for non-flood images
    non_flood_source = base_dir / "Non_Flood_Images"
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    
    # Counters
    flood_count = 0
    non_flood_count = 0
    skipped_count = 0
    
    
    # Copy flood-prone images
    print("\nProcessing flood-prone images...")
    for source_dir in flood_sources:
        if not source_dir.exists():
            print(f"  [SKIP] Not found: {source_dir}")
            continue
        
        print(f"  Processing: {source_dir}")
        image_files = [f for f in source_dir.rglob('*') if f.suffix in image_extensions and f.is_file()]
        
        for img_path in tqdm(image_files, desc=f"    Moving from {source_dir.name}", leave=False):
            try:
                # Generate unique filename to avoid conflicts
                filename = img_path.name
                dest_path = flood_prone_dir / filename
                
                # If file exists, add counter
                counter = 1
                while dest_path.exists():
                    stem = img_path.stem
                    suffix = img_path.suffix
                    dest_path = flood_prone_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                shutil.copy2(img_path, dest_path)
                flood_count += 1
            except Exception as e:
                print(f"    [ERROR] Error copying {img_path}: {e}")
                skipped_count += 1
    
    # Copy non-flood-prone images
    print("\nProcessing non-flood-prone images...")
    if non_flood_source.exists():
        print(f"  Processing: {non_flood_source}")
        image_files = [f for f in non_flood_source.rglob('*') if f.suffix in image_extensions and f.is_file()]
        
        for img_path in tqdm(image_files, desc=f"    Moving from {non_flood_source.name}", leave=False):
            try:
                # Generate unique filename to avoid conflicts
                filename = img_path.name
                dest_path = non_flood_prone_dir / filename
                
                # If file exists, add counter
                counter = 1
                while dest_path.exists():
                    stem = img_path.stem
                    suffix = img_path.suffix
                    dest_path = non_flood_prone_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                shutil.copy2(img_path, dest_path)
                non_flood_count += 1
            except Exception as e:
                print(f"    [ERROR] Error copying {img_path}: {e}")
                skipped_count += 1
    else:
        print(f"  [SKIP] Not found: {non_flood_source}")
    
    # Summary
    print("\n" + "=" * 60)
    print("ORGANIZATION COMPLETE")
    print("=" * 60)
    print(f"[SUCCESS] Flood-prone images: {flood_count}")
    print(f"[SUCCESS] Non-flood-prone images: {non_flood_count}")
    print(f"[WARNING] Skipped/Failed: {skipped_count}")
    print(f"[TOTAL] Total images: {flood_count + non_flood_count}")
    print(f"\nImages organized in:")
    print(f"   - {flood_prone_dir}")
    print(f"   - {non_flood_prone_dir}")
    print("=" * 60)


if __name__ == '__main__':
    organize_images()

