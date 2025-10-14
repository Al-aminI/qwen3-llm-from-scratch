"""
Script to push multiple fine-tuned Qwen models to Hugging Face Hub.
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo
import glob

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

def push_single_file_to_hub(
    file_path: str,
    repo_id: str,
    token: str,
    commit_message: str = "Add model file"
):
    """Push a single model file to Hugging Face Hub."""
    
    print(f"Pushing file {file_path} to {repo_id}...")
    
    login(token=token)
    api = HfApi()
    
    try:
        create_repo(repo_id=repo_id, token=token, exist_ok=True)
        print(f"✅ Repository {repo_id} is ready")
    except Exception as e:
        print(f"⚠️  Repository creation issue: {e}")
    
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=repo_id,
            commit_message=commit_message,
            token=token
        )
        print(f"✅ File successfully pushed to https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Error pushing file: {e}")
        return False

def push_model_to_hub(
    model_path: str,
    repo_id: str,
    token: str,
    commit_message: str = "Add fine-tuned Qwen model"
):
    """Push the fine-tuned model to Hugging Face Hub."""
    
    print(f"Pushing model from {model_path} to {repo_id}...")
    
    login(token=token)
    api = HfApi()
    
    try:
        create_repo(repo_id=repo_id, token=token, exist_ok=True)
        print(f"✅ Repository {repo_id} is ready")
    except Exception as e:
        print(f"⚠️  Repository creation issue: {e}")
    
    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            commit_message=commit_message,
            token=token
        )
        print(f"✅ Model successfully pushed to https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Error pushing model: {e}")
        return False

def get_model_mapping(models_dir: str, username: str):
    """Generate proper repository names for each model."""
    model_mapping = {}
    
    model_patterns = {
        'imdb_sentiment_classifier.pt': f'{username}/qwen3-imdb-sentiment-classifier',
        'lora_sentiment_classifier.pt': f'{username}/qwen3-lora-sentiment-classifier',
        'lora_weights.pt': f'{username}/qwen3-lora-weights',
        'qlora_sentiment_classifier.pt': f'{username}/qwen3-qlora-sentiment-classifier',
        'qlora_weights.pt': f'{username}/qwen3-qlora-weights',
        'best_model1.pt': f'{username}/qwen3-best-model',
        'final_model1.pt': f'{username}/qwen3-final-model',
        'final_model_converted.pt': f'{username}/qwen3-final-model-converted'
    }
    
    for filename, repo_name in model_patterns.items():
        file_path = os.path.join(models_dir, filename)
        if os.path.exists(file_path):
            model_mapping[file_path] = {
                'repo_id': repo_name,
                'filename': filename,
                'description': get_model_description(filename)
            }
    
    return model_mapping

def get_model_description(filename: str):
    """Get description for each model type."""
    descriptions = {
        'imdb_sentiment_classifier.pt': 'Qwen-3 model fine-tuned for IMDB sentiment classification',
        'lora_sentiment_classifier.pt': 'Qwen-3 model with LoRA fine-tuning for sentiment classification',
        'lora_weights.pt': 'LoRA adapter weights for Qwen-3 sentiment classification',
        'qlora_sentiment_classifier.pt': 'Qwen-3 model with QLoRA fine-tuning for sentiment classification',
        'qlora_weights.pt': 'QLoRA adapter weights for Qwen-3 sentiment classification',
        'best_model1.pt': 'Best performing Qwen-3 model checkpoint',
        'final_model1.pt': 'Final Qwen-3 model checkpoint',
        'final_model_converted.pt': 'Converted final Qwen-3 model checkpoint'
    }
    return descriptions.get(filename, 'Qwen-3 fine-tuned model')

def main():
    parser = argparse.ArgumentParser(description='Push multiple fine-tuned Qwen models to Hugging Face Hub')
    parser.add_argument('--models-dir', type=str, default='qwen-llm/models', 
                       help='Directory containing the model files')
    parser.add_argument('--username', type=str, default='AlaminI', 
                       help='Your Hugging Face username')
    parser.add_argument('--token', type=str, default='huggingface token', 
                       help='Hugging Face token')
    parser.add_argument('--single-model', type=str, default=None,
                       help='Push only a specific model file (optional)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be pushed without actually pushing')
    
    args = parser.parse_args()
    
    if not Path(args.models_dir).exists():
        print(f"❌ Models directory {args.models_dir} does not exist!")
        return
    
    # Get model mapping
    model_mapping = get_model_mapping(args.models_dir, args.username)
    
    if not model_mapping:
        print(f"❌ No model files found in {args.models_dir}!")
        return
    
    print(f"📋 Found {len(model_mapping)} models to push:")
    for file_path, info in model_mapping.items():
        print(f"  • {info['filename']} → {info['repo_id']}")
        print(f"    Description: {info['description']}")
    
    if args.dry_run:
        print("\n🔍 Dry run mode - no models will be pushed")
        return
    
    if args.single_model:
        single_file_path = os.path.join(args.models_dir, args.single_model)
        if single_file_path in model_mapping:
            info = model_mapping[single_file_path]
            print(f"\n🚀 Pushing single model: {info['filename']}")
            success = push_single_file_to_hub(
                file_path=single_file_path,
                repo_id=info['repo_id'],
                token=args.token,
                commit_message=f"Add {info['description']}"
            )
            if success:
                print(f"\n🎉 Success! Model available at: https://huggingface.co/{info['repo_id']}")
            else:
                print(f"\n❌ Failed to push {info['filename']}")
        else:
            print(f"❌ Model {args.single_model} not found!")
        return
    
    print(f"\n🚀 Starting to push {len(model_mapping)} models...")
    successful_pushes = []
    failed_pushes = []
    
    for file_path, info in model_mapping.items():
        print(f"\n{'='*60}")
        print(f"Pushing: {info['filename']}")
        print(f"Repository: {info['repo_id']}")
        print(f"{'='*60}")
        
        success = push_single_file_to_hub(
            file_path=file_path,
            repo_id=info['repo_id'],
            token=args.token,
            commit_message=f"Add {info['description']}"
        )
        
        if success:
            successful_pushes.append(info)
        else:
            failed_pushes.append(info)
    
    print(f"\n{'='*60}")
    print(f"📊 PUSH SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully pushed: {len(successful_pushes)} models")
    print(f"❌ Failed to push: {len(failed_pushes)} models")
    
    if successful_pushes:
        print(f"\n🎉 Successfully pushed models:")
        for info in successful_pushes:
            print(f"  • {info['filename']} → https://huggingface.co/{info['repo_id']}")
    
    if failed_pushes:
        print(f"\n❌ Failed to push:")
        for info in failed_pushes:
            print(f"  • {info['filename']} → {info['repo_id']}")
    
    if successful_pushes:
        print(f"\n💡 Usage examples:")
        print(f"# For LoRA models:")
        print(f"from transformers import AutoTokenizer, AutoModelForCausalLM")
        print(f"from peft import PeftModel")
        print(f"")
        print(f"base_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B')")
        print(f"model = PeftModel.from_pretrained(base_model, 'AlaminI/qwen3-lora-sentiment-classifier')")
        print(f"tokenizer = AutoTokenizer.from_pretrained('AlaminI/qwen3-lora-sentiment-classifier')")

if __name__ == "__main__":
    main()
