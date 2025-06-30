#!/usr/bin/env python3
"""
Script to add .gitkeep files to all subdirectories in the data folder.
This ensures that empty directories are tracked by git even when the data folder is in .gitignore.
"""

import os
from pathlib import Path

def add_gitkeep_to_directories(root_path: str | Path):
    """
    Recursively add .gitkeep files to all subdirectories.
    
    Args:
        root_path: Path to the root directory to process
    """
    root = Path(root_path)
    
    if not root.exists():
        print(f"Error: {root_path} does not exist")
        return
    
    if not root.is_dir():
        print(f"Error: {root_path} is not a directory")
        return
    
    gitkeep_count = 0
    
    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)
        
        # Skip if .gitkeep already exists
        gitkeep_file = dir_path / ".gitkeep"
        if gitkeep_file.exists():
            print(f"  Skipping {dir_path} (already has .gitkeep)")
            continue
        
        # Add .gitkeep file
        gitkeep_file.touch()
        print(f"  Added .gitkeep to {dir_path}")
        gitkeep_count += 1
    
    print(f"\nTotal .gitkeep files added: {gitkeep_count}")

if __name__ == "__main__":
    # Add .gitkeep to all subdirectories in the data folder
    data_path = Path("data")
    
    print(f"Adding .gitkeep files to subdirectories in {data_path.absolute()}")
    print("=" * 60)
    
    add_gitkeep_to_directories(data_path)
    
    print("\nDone!") 