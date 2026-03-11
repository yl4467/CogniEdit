import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def organize_images_from_jsonl(jsonl_path, output_dir="/home/ma-user/work/yanli/test_yl/All_Data_Checkpoints"):
    """
    Read JSONL file and organize images into separate folders
    
    Args:
        jsonl_path: Path to the JSONL file
        output_dir: Base directory for organized images
    """
    # Create output directories
    original_dir = os.path.join(output_dir, "original_images")
    target_dir = os.path.join(output_dir, "target_images")
    
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    
    # Track statistics
    stats = {
        "total_entries": 0,
        "copied_originals": 0,
        "copied_targets": 0,
        "missing_originals": 0,
        "missing_targets": 0,
        "errors": []
    }
    
    # Read JSONL and copy images
    print(f"Reading {jsonl_path}...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    root_path = '/home/ma-user/work/yanli/test_yl/Omni_Pos_Data'
    print(f"Processing {len(lines)} entries...")
    for line in tqdm(lines):
        try:
            entry = json.loads(line.strip())
            stats["total_entries"] += 1
            
            # Get image paths
            source_image = os.path.join(root_path, entry.get("source_image"))
            target_image = os.path.join(root_path, entry.get("target_image"))
            
            # Copy source image
            if source_image and os.path.exists(source_image):
                dest_path = os.path.join(original_dir, os.path.basename(source_image))
                shutil.copy2(source_image, dest_path)
                stats["copied_originals"] += 1
            elif source_image:
                stats["missing_originals"] += 1
                stats["errors"].append(f"Missing source: {source_image}")
            
            # Copy target image
            if target_image and os.path.exists(target_image):
                dest_path = os.path.join(target_dir, os.path.basename(target_image))
                shutil.copy2(target_image, dest_path)
                stats["copied_targets"] += 1
            elif target_image:
                stats["missing_targets"] += 1
                stats["errors"].append(f"Missing target: {target_image}")
                
        except Exception as e:
            stats["errors"].append(f"Error processing line: {str(e)}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total entries processed: {stats['total_entries']}")
    print(f"Original images copied: {stats['copied_originals']}")
    print(f"Target images copied: {stats['copied_targets']}")
    print(f"Missing original images: {stats['missing_originals']}")
    print(f"Missing target images: {stats['missing_targets']}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    if stats["errors"]:
        print(f"\nErrors encountered: {len(stats['errors'])}")
        print("First 10 errors:")
        for error in stats["errors"][:10]:
            print(f"  - {error}")
    
    return stats

if __name__ == "__main__":
    import sys
    
    # Default to filtered_position_only.jsonl
    jsonl_file = "/home/ma-user/work/yanli/test_yl/Omni_Pos_Data/test_select_pos.jsonl"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        jsonl_file = sys.argv[1]
    
    if not os.path.exists(jsonl_file):
        print(f"Error: File '{jsonl_file}' not found!")
        print("\nAvailable JSONL files:")
        for f in Path(".").glob("**/*.jsonl"):
            print(f"  - {f}")
        sys.exit(1)
    
    # Run the organization
    organize_images_from_jsonl(jsonl_file)
    print("\nDone! Images organized successfully.")