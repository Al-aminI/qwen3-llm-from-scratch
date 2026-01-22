"""
🎯 MULTIMODAL PRETRAINING TRAINER

This module contains the complete training pipeline for multimodal pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import time
from tqdm import tqdm
from typing import Dict, Any, Optional

from ..model.minimal_llm import MultimodalLLM
from .optimizer import setup_muon_optimizer

def compute_weighted_loss_eval(logits, labels, weights, config):
    """Compute weighted loss for multimodal evaluation"""
    # Flatten all tensors
    logits_flat = logits.view(-1, config.vocab_size)
    labels_flat = labels.view(-1)
    weights_flat = weights.view(-1)
    
    # Compute cross entropy loss
    loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=config.pad_token_id, reduction='none')
    
    # Apply weights
    weighted_loss = loss * weights_flat
    
    # Return mean loss
    return weighted_loss.mean()

def evaluate_multimodal_model(model: nn.Module, val_loader: DataLoader, config):
    """
    📊 MULTIMODAL MODEL EVALUATION FUNCTION
    
    This function evaluates the model's performance on validation data:
    
    🎯 Metrics Computed:
    1. Loss: Cross-entropy loss (lower is better)
    2. Accuracy: Percentage of correct next-token predictions
    3. Perplexity: exp(loss) - measures model's "surprise" (lower is better)
    4. Audio Loss: Loss specifically on audio tokens
    5. Text Loss: Loss specifically on text tokens
    
    🔍 Why these metrics matter:
    - Loss: Direct measure of how well the model predicts
    - Accuracy: Human-interpretable measure of correctness
    - Perplexity: How "confused" the model is (lower = more confident)
    - Audio/Text Loss: Separate evaluation of different modalities
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    audio_loss = 0
    text_loss = 0
    audio_tokens = 0
    text_tokens = 0

    device = next(model.parameters()).device

    with torch.no_grad():  # Disable gradients for evaluation
        for i, batch in enumerate(val_loader):
            if i >= config.eval_steps:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            weights = batch['weights'].to(device)

            # Forward pass with mixed precision
            with autocast(enabled=config.use_amp):
                logits = model(input_ids, attention_mask=attention_mask)
                # Use weighted loss for multimodal evaluation
                loss = compute_weighted_loss_eval(logits, labels, weights, config)

            # Accumulate metrics
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

            # Compute accuracy
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == labels).sum().item()

            # Separate audio and text losses using weights (exclude padding tokens)
            valid_mask = (attention_mask == 1) & (weights > 0)  # Only non-padding, non-zero weight tokens
            audio_mask = (weights == config.audio_weight) & valid_mask
            text_mask = (weights == config.text_weight) & valid_mask
            
            if audio_mask.any():
                # Compute loss only on audio tokens
                audio_logits = logits.view(-1, config.vocab_size)[audio_mask.view(-1)]
                audio_labels = labels.view(-1)[audio_mask.view(-1)]
                
                audio_batch_loss = F.cross_entropy(
                    audio_logits, 
                    audio_labels, 
                    ignore_index=config.pad_token_id
                )
                
                # Check for NaN
                if torch.isnan(audio_batch_loss):
                    continue
                
                audio_loss += audio_batch_loss.item() * audio_mask.sum().item()
                audio_tokens += audio_mask.sum().item()
            
            if text_mask.any():
                # Compute loss only on text tokens
                text_logits = logits.view(-1, config.vocab_size)[text_mask.view(-1)]
                text_labels = labels.view(-1)[text_mask.view(-1)]
                text_loss += F.cross_entropy(
                    text_logits, 
                    text_labels, 
                    ignore_index=config.pad_token_id
                ).item() * text_mask.sum().item()
                text_tokens += text_mask.sum().item()

    # Compute final metrics
    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))  # Cap to prevent overflow
    
    avg_audio_loss = audio_loss / audio_tokens if audio_tokens > 0 else 0
    avg_text_loss = text_loss / text_tokens if text_tokens > 0 else 0

    model.train()
    return {
        'val_loss': avg_loss, 
        'val_accuracy': accuracy, 
        'val_perplexity': perplexity,
        'val_audio_loss': avg_audio_loss,
        'val_text_loss': avg_text_loss
    }

def load_multimodal_checkpoint(model_path: str, config):
    """
    📦 LOAD CHECKPOINT FOR RESUMING MULTIMODAL TRAINING
    
    This function loads a previously trained multimodal model checkpoint and returns
    the model, optimizers, schedulers, and training state.
    """
    print(f"📦 Loading multimodal checkpoint from {model_path}")
    
    # Load checkpoint with safe loading to handle import path changes
    try:
        # Try loading with weights_only=False to handle custom classes
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"⚠️ Error loading checkpoint: {e}")
        print("🔄 Trying alternative loading method...")
        try:
            # Try with weights_only=True to avoid custom class issues
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        except Exception as e2:
            print(f"⚠️ Alternative loading also failed: {e2}")
            print("🔄 Trying to load only the model state dict...")
            # Last resort: try to extract just the model state dict
            import zipfile
            import io
            with zipfile.ZipFile(model_path, 'r') as zip_file:
                # Look for the model state dict in the zip file
                for name in zip_file.namelist():
                    if 'model_state_dict' in name:
                        with zip_file.open(name) as f:
                            model_data = torch.load(io.BytesIO(f.read()), map_location='cpu')
                            checkpoint = {'model_state_dict': model_data}
                            break
                else:
                    raise Exception("Could not find model state dict in checkpoint")
    
    # Create model with current config (ignore old config from checkpoint)
    model = MultimodalLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Extract training state
    start_step = checkpoint.get('step', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"✅ Multimodal checkpoint loaded successfully")
    print(f"   Previous training step: {start_step}")
    print(f"   Previous best val loss: {best_val_loss:.4f}")
    
    return model, start_step, best_val_loss

class MultimodalPretrainingTrainer:
    """
    🎯 MULTIMODAL PRETRAINING TRAINER
    
    Complete training pipeline for multimodal pretraining with text and audio.
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def setup_model(self):
        """Setup the multimodal model for training."""
        print("🏗️ Setting up multimodal model...")
        
        self.model = MultimodalLLM(self.config)
        
        # Resize model embeddings to accommodate SNAC tokens (like tts-snac-base.py)
        if hasattr(self.model, 'token_embedding'):
            # The config already has the correct vocab size for SNAC tokens
            new_vocab_size = self.config.vocab_size  # 50304
            self.model.token_embedding = torch.nn.Embedding(new_vocab_size, self.config.d_model)
            self.model.lm_head = torch.nn.Linear(self.config.d_model, new_vocab_size, bias=False)
            print(f"  🔧 Resized embeddings to vocab size: {new_vocab_size} (GPT-2 + SNAC tokens)")
        
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  📊 Total parameters: {total_params:,}")
        
        return self.model
    
    def setup_optimizers(self):
        """Setup single Muon optimizer and scheduler."""
        print("🚀 Setting up optimizer...")
        
        # Get the Muon optimizer (first one from the list)
        optimizers = setup_muon_optimizer(self.model, self.config)
        self.optimizer = optimizers[0]  # Use only the Muon optimizer
        
        # Single scheduler
        warmup_steps = self.config.max_steps // 20  # 5% warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps  # Linear warmup
            else:
                progress = (step - warmup_steps) / (self.config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))  # Cosine decay

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.config.use_amp else None
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Total parameters: {total_params:,}")
        
        return self.optimizer, self.scheduler
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, resume_from: Optional[str] = None):
        """
        🔄 COMPLETE MULTIMODAL TRAINING LOOP
        
        This is the heart of the multimodal training process, implementing:
        
        🎯 Key Features:
        1. Gradient Accumulation: Simulate larger batch sizes
        2. Mixed Precision: Faster training with minimal accuracy loss
        3. Learning Rate Scheduling: Warmup + cosine decay
        4. Gradient Clipping: Prevent gradient explosions
        5. Model Checkpointing: Save best model
        6. Progress Tracking: Real-time metrics
        7. Multimodal Loss Weighting: Different weights for text vs audio
        
        📊 Training Process:
        1. Forward pass: Compute predictions and loss
        2. Backward pass: Compute gradients
        3. Gradient accumulation: Accumulate gradients over multiple steps
        4. Optimizer step: Update parameters
        5. Learning rate scheduling: Adjust learning rate
        6. Evaluation: Check performance on validation set
        7. Checkpointing: Save best model
        """
        print(f"\n🚀 Training Multimodal Qwen3 model with Muon optimizer")
        
        # Setup model and optimizers
        if resume_from:
            print(f"🔄 Resuming training from {resume_from}...")
            self.model, start_step, best_val_loss = load_multimodal_checkpoint(resume_from, self.config)
        else:
            self.setup_model()
            start_step = 0
            best_val_loss = float('inf')
        
        self.setup_optimizers()
        
        # Training loop
        self.model.train()
        step = start_step
        start_time = time.time()

        pbar = tqdm(total=self.config.max_steps, initial=start_step, desc="Multimodal Training")

        while step < self.config.max_steps:
            for batch_idx, batch in enumerate(train_loader):
                if step >= self.config.max_steps:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                weights = batch['weights'].to(self.device)

                # Forward pass with gradient accumulation
                if self.config.use_amp:
                    with autocast():
                        logits = self.model(input_ids, attention_mask=attention_mask)
                        # Use weighted loss for multimodal training
                        loss = self.compute_weighted_loss(logits, labels, weights)
                        loss = loss / self.config.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                else:
                    logits = self.model(input_ids, attention_mask=attention_mask)
                    # Use weighted loss for multimodal training
                    loss = self.compute_weighted_loss(logits, labels, weights)
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()

                # Optimizer step after accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.optimizer.zero_grad()
                        self.scheduler.step()
                        self.scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.scheduler.step()

                # Logging
                if step % 10 == 0:
                    with torch.no_grad():
                        predictions = logits.argmax(dim=-1)
                        accuracy = (predictions == labels).float().mean().item()
                        current_loss = loss.item() * self.config.gradient_accumulation_steps
                        perplexity = math.exp(min(current_loss, 20))

                    pbar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'acc': f'{accuracy:.3f}',
                        'ppl': f'{perplexity:.1f}',
                        'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                    })

                # Evaluation
                if step % self.config.eval_every == 0 and step > 0:
                    eval_metrics = evaluate_multimodal_model(self.model, val_loader, self.config)
                    print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                          f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                          f"Val PPL: {eval_metrics['val_perplexity']:.2f}")
                    print(f"  Audio Loss: {eval_metrics['val_audio_loss']:.4f}, "
                          f"Text Loss: {eval_metrics['val_text_loss']:.4f}")

                    if eval_metrics['val_loss'] < best_val_loss:
                        best_val_loss = eval_metrics['val_loss']
                        # Save best model
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'config': self.config,
                            'step': step,
                            'best_val_loss': best_val_loss,
                            'final_metrics': eval_metrics
                        }, 'models/best_multimodal_model.pt')
                        print(f"💾 Saved best multimodal model with val_loss: {best_val_loss:.4f}")

                step += 1
                if step % 10 == 0:
                    pbar.update(10)

        pbar.close()

        training_time = time.time() - start_time
        print(f"  ⏱️ Multimodal training completed in {training_time:.1f} seconds")

        # Final evaluation
        final_eval = evaluate_multimodal_model(self.model, val_loader, self.config)
        print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, "
              f"Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")
        print(f"  🎵 Audio Loss: {final_eval['val_audio_loss']:.4f}, "
              f"📝 Text Loss: {final_eval['val_text_loss']:.4f}")

        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'step': step,
            'final_metrics': final_eval
        }, 'models/final_multimodal_model.pt')
        print(f"💾 Saved final multimodal model to final_multimodal_model.pt")

        return self.model, final_eval
    
    def compute_weighted_loss(self, logits, labels, weights):
        """Compute weighted loss for multimodal training"""
        # Flatten all tensors
        logits_flat = logits.view(-1, self.config.vocab_size)
        labels_flat = labels.view(-1)
        weights_flat = weights.view(-1)
        
        # Compute cross entropy loss
        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=self.config.pad_token_id, reduction='none')
        
        # Apply weights
        weighted_loss = loss * weights_flat
        
        # Return mean loss
        return weighted_loss.mean()
