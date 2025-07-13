#!/usr/bin/env python3
"""
Script to upload AdaFortiTran repository to Hugging Face.
This script prepares the repository for Hugging Face upload with minimal changes.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

def check_huggingface_hub_installed() -> bool:
    """Check if huggingface_hub is installed."""
    try:
        import huggingface_hub
        return True
    except ImportError:
        return False


def create_huggingface_files(repo_path: Path):
    """Create necessary files for Hugging Face upload."""
    
    # Create .gitattributes file for large files
    gitattributes_content = """*.mat filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
"""
    
    gitattributes_path = repo_path / ".gitattributes"
    if not gitattributes_path.exists():
        with open(gitattributes_path, 'w') as f:
            f.write(gitattributes_content)
        print(f"  Created {gitattributes_path}")
    
    # Create .huggingfaceignore file
    huggingfaceignore_content = """# Ignore large data files during upload
data/train/
data/val/
data/test/

# Ignore model checkpoints and logs
*.ckpt
*.pth
*.pt
logs/
runs/
checkpoints/

# Ignore temporary files
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.DS_Store
Thumbs.db

# Ignore IDE files
.vscode/
.idea/
*.swp
*.swo

# Ignore environment files
.env
.venv/
venv/
env/
"""
    
    huggingfaceignore_path = repo_path / ".huggingfaceignore"
    if not huggingfaceignore_path.exists():
        with open(huggingfaceignore_path, 'w') as f:
            f.write(huggingfaceignore_content)
        print(f"  Created {huggingfaceignore_path}")

def create_model_card(repo_path: Path):
    """Create a model card for Hugging Face."""
    
    model_card_content = """---
language:
- en
tags:
- pytorch
- transformer
- channel-estimation
- ofdm
- wireless
- adaptive
license: mit
datasets:
- custom
metrics:
- mse
---

# AdaFortiTran: Adaptive Transformer Model for Robust OFDM Channel Estimation

## Model Description

AdaFortiTran is a novel adaptive transformer-based model for OFDM channel estimation that dynamically adapts to varying channel conditions (SNR, delay spread, Doppler shift). The model combines the power of transformer architectures with channel-aware adaptation mechanisms to achieve robust performance across diverse wireless environments.

## Key Features

- **🔄 Adaptive Architecture**: Dynamically adapts to channel conditions using meta-information
- **⚡ High Performance**: State-of-the-art results on OFDM channel estimation tasks
- **🧠 Transformer-Based**: Leverages attention mechanisms for long-range dependencies
- **🎯 Robust**: Maintains performance across varying SNR, delay spread, and Doppler conditions
- **🚀 Production Ready**: Comprehensive training pipeline with advanced features

## Architecture

The project implements three model variants:

1. **Linear Estimator**: Simple learned linear transformation baseline
2. **FortiTran**: Fixed transformer-based channel estimator
3. **AdaFortiTran**: Adaptive transformer with channel condition awareness

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python src/main.py \
    --model_name adafortitran \
    --system_config_path config/system_config.yaml \
    --model_config_path config/adafortitran.yaml \
    --train_set data/train \
    --val_set data/val \
    --test_set data/test \
    --exp_id my_experiment
```

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{guler2025adafortitranadaptivetransformermodel,
      title={AdaFortiTran: An Adaptive Transformer Model for Robust OFDM Channel Estimation}, 
      author={Berkay Guler and Hamid Jafarkhani},
      year={2025},
      eprint={2505.09076},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.09076}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
"""
    
    model_card_path = repo_path / "README.md"
    if model_card_path.exists():
        # Backup original README
        backup_path = repo_path / "README_original.md"
        if not backup_path.exists():
            shutil.copy2(model_card_path, backup_path)
            print(f"  Backed up original README to {backup_path}")
    
    with open(model_card_path, 'w') as f:
        f.write(model_card_content)
    print(f"  Updated {model_card_path} for Hugging Face")

def cleanup_generated_files(repo_path: Path):
    """Remove files generated for Hugging Face upload."""
    print("\nCleaning up generated files...")
    
    files_to_remove = [
        ".gitattributes",
        ".huggingfaceignore"
    ]
    
    for file_name in files_to_remove:
        file_path = repo_path / file_name
        if file_path.exists():
            file_path.unlink()
            print(f"  Removed {file_path}")
    
    # Restore original README if backup exists
    backup_path = repo_path / "README_original.md"
    readme_path = repo_path / "README.md"
    
    if backup_path.exists():
        shutil.copy2(backup_path, readme_path)
        backup_path.unlink()
        print(f"  Restored original README.md")
    
    # Remove git remote if it was added
    try:
        result = subprocess.run(["git", "remote", "get-url", "origin"], 
                              capture_output=True, text=True, check=False)
        if "huggingface.co" in result.stdout:
            subprocess.run(["git", "remote", "remove", "origin"], check=False)
            print("  Removed Hugging Face remote")
    except Exception:
        pass
    
    print("✅ Cleanup completed")

def upload_to_huggingface(repo_path: Path, repo_name: str, private: bool = False):
    """Upload the repository to Hugging Face."""
    
    try:
        from huggingface_hub import HfApi, create_repo
        
        api = HfApi()
        
        # Check if user is logged in
        try:
            user_info = api.whoami()
            username = user_info['name']
            print(f"✅ Logged in as: {username}")
        except Exception:
            print("❌ Not logged in to Hugging Face")
            print("Please run: huggingface-cli login")
            return False
        
        # Create repository
        repo_id = f"{username}/{repo_name}"
        print(f"Creating repository: {repo_id}")
        
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True
            )
            print(f"✅ Repository created/updated: {repo_id}")
        except Exception as e:
            print(f"❌ Failed to create repository: {e}")
            return False
        
        # Upload files
        print("Uploading files to Hugging Face...")
        
        # Use git to push to Hugging Face
        os.chdir(repo_path)
        
        # Initialize git if not already done
        if not (repo_path / ".git").exists():
            subprocess.run(["git", "init"], check=True)
            print("  Initialized git repository")
        
        # Add all files
        subprocess.run(["git", "add", "."], check=True)
        print("  Added files to git")
        
        # Check if there are changes to commit
        result = subprocess.run(["git", "status", "--porcelain"], 
                              capture_output=True, text=True, check=True)
        
        if result.stdout.strip():
            # There are changes to commit
            subprocess.run(["git", "commit", "-m", "Initial commit for Hugging Face"], check=True)
            print("  Committed changes")
        else:
            # No changes to commit
            print("  No changes to commit (working tree clean)")
        
        # Get Hugging Face token for authentication
        try:
            token = api.token
            if not token:
                print("❌ No Hugging Face token found")
                print("Please run: huggingface-cli login")
                return False
        except Exception:
            print("❌ Failed to get Hugging Face token")
            print("Please run: huggingface-cli login")
            return False
        
        # Add Hugging Face remote with token authentication
        remote_url = f"https://{username}:{token}@huggingface.co/{repo_id}"
        
        # Check if origin remote already exists
        result = subprocess.run(["git", "remote", "get-url", "origin"], 
                              capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            # Remote exists, update it
            subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True)
            print(f"  Updated remote: {repo_id}")
        else:
            # Remote doesn't exist, add it
            subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
            print(f"  Added remote: {repo_id}")
        
        # Get current branch name
        result = subprocess.run(["git", "branch", "--show-current"], 
                              capture_output=True, text=True, check=True)
        current_branch = result.stdout.strip()
        
        # Push to Hugging Face using current branch
        subprocess.run(["git", "push", "-u", "origin", current_branch], check=True)
        print(f"  Pushed to Hugging Face (branch: {current_branch})")
        
        print(f"\n🎉 Successfully uploaded to: https://huggingface.co/{repo_id}")
        return True
        
    except ImportError:
        print("❌ huggingface_hub not available")
        return False
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    """Main function to handle the upload process."""
    
    repo_path = Path.cwd()
    print(f"Preparing AdaFortiTran repository for Hugging Face upload")
    print(f"Repository path: {repo_path.absolute()}")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not (repo_path / "src" / "models" / "adafortitran.py").exists():
        print("❌ Error: Please run this script from the AdaFortiTran root directory")
        return
    
    # Check if huggingface_hub is installed
    if not check_huggingface_hub_installed():
        print("❌ huggingface_hub is not installed")
        print("Please install it manually: pip install huggingface_hub")
        return
    
    # Get repository name
    repo_name = input("Enter repository name for Hugging Face (default: adafortitran): ").strip()
    if not repo_name:
        repo_name = "adafortitran"
    
    # Ask for private/public
    private_input = input("Make repository private? (y/N): ").strip().lower()
    private = private_input in ['y', 'yes']
    
    print("\nPreparing repository...")
    
    # Create necessary files
    create_huggingface_files(repo_path)
    create_model_card(repo_path)
    
    print("\nUploading to Hugging Face...")
    
    # Upload to Hugging Face
    if upload_to_huggingface(repo_path, repo_name, private):
        print("\n✅ Upload completed successfully!")
        print(f"🔗 View your repository at: https://huggingface.co/{repo_name}")
        
        # Ask user if they want to cleanup
        cleanup_input = input("\nRemove generated files and restore original state? (Y/n): ").strip().lower()
        if cleanup_input not in ['n', 'no']:
            cleanup_generated_files(repo_path)
        else:
            print("Generated files kept for future uploads")
    else:
        print("\n❌ Upload failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 