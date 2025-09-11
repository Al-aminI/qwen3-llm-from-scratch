"""
🎯 PRETRAINING TRAINER

This module contains the complete training pipeline for pretraining.
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

from ..model.minimal_llm import MinimalLLM
from .optimizer import setup_muon_optimizer

def evaluate_model(model: nn.Module, val_loader: DataLoader, config):
    """
    📊 MODEL EVALUATION FUNCTION
    
    This function evaluates the model's performance on validation data:
    
    🎯 Metrics Computed:
    1. Loss: Cross-entropy loss (lower is better)
    2. Accuracy: Percentage of correct next-token predictions
    3. Perplexity: exp(loss) - measures model's "surprise" (lower is better)
    
    🔍 Why these metrics matter:
    - Loss: Direct measure of how well the model predicts
    - Accuracy: Human-interpretable measure of correctness
    - Perplexity: How "confused" the model is (lower = more confident)
    
    📈 Good values:
    - Loss: 2-4 for language models
    - Accuracy: 0.3-0.5 (30-50% correct predictions)
    - Perplexity: 10-50 (depends on vocabulary size)
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():  # Disable gradients for evaluation
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass with mixed precision
            with autocast(enabled=config.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            # Accumulate metrics
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            # Compute accuracy
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    # Compute final metrics
    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))  # Cap to prevent overflow

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def load_checkpoint(model_path: str, config):
    """
    📦 LOAD CHECKPOINT FOR RESUMING TRAINING
    
    This function loads a previously trained model checkpoint and returns
    the model, optimizers, schedulers, and training state.
    """
    print(f"📦 Loading checkpoint from {model_path}")
    
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
    model = MinimalLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Extract training state
    start_step = checkpoint.get('step', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"✅ Checkpoint loaded successfully")
    print(f"   Previous training step: {start_step}")
    print(f"   Previous best val loss: {best_val_loss:.4f}")
    
    return model, start_step, best_val_loss

class PretrainingTrainer:
    """
    🎯 PRETRAINING TRAINER
    
    Complete training pipeline for pretraining language models.
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizers = None
        self.schedulers = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def setup_model(self):
        """Setup the model for training."""
        print("🏗️ Setting up model...")
        
        self.model = MinimalLLM(self.config)
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  📊 Total parameters: {total_params:,}")
        
        return self.model
    
    def setup_optimizers(self):
        """Setup optimizers and schedulers."""
        print("🚀 Setting up optimizers...")
        
        # Setup optimizers
        self.optimizers = setup_muon_optimizer(self.model, self.config)
        
        # Learning rate scheduling
        self.schedulers = []
        for optimizer in self.optimizers:
            warmup_steps = self.config.max_steps // 20  # 5% warmup
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps  # Linear warmup
                else:
                    progress = (step - warmup_steps) / (self.config.max_steps - warmup_steps)
                    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))  # Cosine decay

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            self.schedulers.append(scheduler)
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.config.use_amp else None
        
        return self.optimizers, self.schedulers
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, resume_from: Optional[str] = None):
        """
        🔄 COMPLETE TRAINING LOOP
        
        This is the heart of the training process, implementing:
        
        🎯 Key Features:
        1. Gradient Accumulation: Simulate larger batch sizes
        2. Mixed Precision: Faster training with minimal accuracy loss
        3. Learning Rate Scheduling: Warmup + cosine decay
        4. Gradient Clipping: Prevent gradient explosions
        5. Model Checkpointing: Save best model
        6. Progress Tracking: Real-time metrics
        
        📊 Training Process:
        1. Forward pass: Compute predictions and loss
        2. Backward pass: Compute gradients
        3. Gradient accumulation: Accumulate gradients over multiple steps
        4. Optimizer step: Update parameters
        5. Learning rate scheduling: Adjust learning rate
        6. Evaluation: Check performance on validation set
        7. Checkpointing: Save best model
        """
        print(f"\n🚀 Training Small Qwen3 model with Muon optimizer")
        
        # Setup model and optimizers
        if resume_from:
            print(f"🔄 Resuming training from {resume_from}...")
            self.model, start_step, best_val_loss = load_checkpoint(resume_from, self.config)
        else:
            self.setup_model()
            start_step = 0
            best_val_loss = float('inf')
        
        self.setup_optimizers()
        
        # Training loop
        self.model.train()
        step = start_step
        start_time = time.time()

        pbar = tqdm(total=self.config.max_steps, initial=start_step, desc="Training")

        while step < self.config.max_steps:
            for batch_idx, (x, y) in enumerate(train_loader):
                if step >= self.config.max_steps:
                    break

                x, y = x.to(self.device), y.to(self.device)

                # Forward pass with gradient accumulation
                if self.config.use_amp:
                    with autocast():
                        logits = self.model(x)
                        loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                        loss = loss / self.config.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                else:
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()

                # Optimizer step after accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.use_amp:
                        for optimizer in self.optimizers:
                            self.scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                        for optimizer in self.optimizers:
                            self.scaler.step(optimizer)
                            optimizer.zero_grad()
                        for scheduler in self.schedulers:
                            scheduler.step()
                        self.scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                        for optimizer in self.optimizers:
                            optimizer.step()
                            optimizer.zero_grad()
                        for scheduler in self.schedulers:
                            scheduler.step()

                # Logging
                if step % 10 == 0:
                    with torch.no_grad():
                        predictions = logits.argmax(dim=-1)
                        accuracy = (predictions == y).float().mean().item()
                        current_loss = loss.item() * self.config.gradient_accumulation_steps
                        perplexity = math.exp(min(current_loss, 20))

                    pbar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'acc': f'{accuracy:.3f}',
                        'ppl': f'{perplexity:.1f}',
                        'lr': f'{self.optimizers[0].param_groups[0]["lr"]:.2e}'
                    })

                # Evaluation
                if step % self.config.eval_every == 0 and step > 0:
                    eval_metrics = evaluate_model(self.model, val_loader, self.config)
                    print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                          f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                          f"Val PPL: {eval_metrics['val_perplexity']:.2f}")

                    if eval_metrics['val_loss'] < best_val_loss:
                        best_val_loss = eval_metrics['val_loss']
                        # Save best model
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'config': self.config,
                            'step': step,
                            'best_val_loss': best_val_loss,
                            'final_metrics': eval_metrics
                        }, 'models/best_model1.pt')
                        print(f"💾 Saved best model with val_loss: {best_val_loss:.4f}")

                step += 1
                if step % 10 == 0:
                    pbar.update(10)

        pbar.close()

        training_time = time.time() - start_time
        print(f"  ⏱️ Training completed in {training_time:.1f} seconds")

        # Final evaluation
        final_eval = evaluate_model(self.model, val_loader, self.config)
        print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, "
              f"Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")

        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'step': step,
            'final_metrics': final_eval
        }, 'models/final_model1.pt')
        print(f"💾 Saved final model to final_model1.pt")

        return self.model, final_eval
