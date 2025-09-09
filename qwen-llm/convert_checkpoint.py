#!/usr/bin/env python3
"""
🔄 CHECKPOINT CONVERTER

This script converts old checkpoints to the new format by extracting
only the model state dict and training state, avoiding import issues.
"""

import torch
import sys
import os

def convert_checkpoint(old_path: str, new_path: str):
    """
    Convert old checkpoint to new format
    """
    print(f"🔄 Converting checkpoint from {old_path} to {new_path}")
    
    try:
        # Try to load the old checkpoint
        print("📦 Loading old checkpoint...")
        checkpoint = torch.load(old_path, map_location='cpu', weights_only=True)
        
        # Extract only the essential data
        new_checkpoint = {
            'model_state_dict': checkpoint['model_state_dict'],
            'step': checkpoint.get('step', 0),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
            'final_metrics': checkpoint.get('final_metrics', {})
        }
        
        # Save the new checkpoint
        print("💾 Saving converted checkpoint...")
        torch.save(new_checkpoint, new_path)
        
        print(f"✅ Checkpoint converted successfully!")
        print(f"   Original step: {new_checkpoint['step']}")
        print(f"   Best val loss: {new_checkpoint['best_val_loss']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting checkpoint: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_checkpoint.py <old_checkpoint> <new_checkpoint>")
        sys.exit(1)
    
    old_path = sys.argv[1]
    new_path = sys.argv[2]
    
    if not os.path.exists(old_path):
        print(f"❌ Old checkpoint not found: {old_path}")
        sys.exit(1)
    
    success = convert_checkpoint(old_path, new_path)
    sys.exit(0 if success else 1)
