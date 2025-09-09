# -*- coding: utf-8 -*-
"""
🏗️ COMPLETE QWEN3 MODEL AND TRAINING PIPELINE

This file contains the complete language model and training infrastructure
with detailed explanations of each component.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import time
from tqdm import tqdm
from typing import List, Optional
from dataclasses import dataclass

# Import our custom components
from config.qwen3_small_config import SmallModelConfig, set_seed, load_and_cache_data, TextTokenDataset, Muon
from qwen3_core_components import Qwen3Attention, SwiGLUFeedForward, TransformerBlock, RMSNorm, Rotary, repeat_kv

# =============================================================================
# 🏗️ COMPONENT 6: COMPLETE LANGUAGE MODEL
# =============================================================================

class MinimalLLM(nn.Module):
    """
    🏗️ COMPLETE QWEN3-STYLE LANGUAGE MODEL
    
    This is the full language model that combines all components:
    
    🧠 Architecture:
    1. Token Embedding: Convert token IDs to vectors
    2. Positional Dropout: Prevent overfitting on positions
    3. Transformer Blocks: Stack of attention + feed-forward layers
    4. Final Normalization: RMSNorm before output
    5. Language Modeling Head: Convert vectors back to token probabilities
    6. Weight Tying: Share weights between input and output embeddings
    
    🎯 Key Features:
    - Pre-norm architecture (more stable training)
    - Weight tying (reduces parameters, improves generalization)
    - Proper initialization (Xavier/He initialization)
    - Efficient memory usage (GQA, RMSNorm)
    
    📊 Parameter Efficiency:
    - Weight tying: Input and output embeddings share weights
    - GQA: Reduces attention memory by 50-75%
    - RMSNorm: More efficient than LayerNorm
    - SwiGLU: More expressive than ReLU with similar cost
    """
    
    def __init__(self, config: SmallModelConfig):
        super().__init__()
        self.config = config

        # Token embedding: converts token IDs to vectors
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional dropout: prevents overfitting on position information
        self.position_dropout = nn.Dropout(config.dropout)

        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final normalization before output
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.output_dropout = nn.Dropout(config.dropout)

        # Language modeling head: converts vectors to token probabilities
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 🔑 WEIGHT TYING: Share weights between input and output embeddings
        # This reduces parameters and improves generalization
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        🎯 PROPER WEIGHT INITIALIZATION
        
        Good initialization is crucial for training stability:
        - Linear layers: Normal distribution with std=0.02
        - Embeddings: Normal distribution with std=0.02
        - Biases: Zero initialization
        
        This follows the initialization scheme used in modern LLMs.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Forward pass through the complete model
        
        Args:
            x: Input token IDs (batch, seq_len)
            
        Returns:
            Logits for next token prediction (batch, seq_len, vocab_size)
        """
        # 1. Convert token IDs to embeddings
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        
        # 2. Apply positional dropout
        x = self.position_dropout(x)

        # 3. Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # 4. Final normalization
        x = self.norm(x)
        x = self.output_dropout(x)
        
        # 5. Convert to token probabilities
        logits = self.lm_head(x)
        return logits

# =============================================================================
# 📊 COMPONENT 7: EVALUATION FUNCTION
# =============================================================================

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: SmallModelConfig):
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

# =============================================================================
# 🚀 COMPONENT 8: OPTIMIZER SETUP
# =============================================================================

def setup_muon_optimizer(model: nn.Module, config: SmallModelConfig):
    """
    🚀 HYBRID OPTIMIZER SETUP
    
    This function sets up a hybrid optimization strategy:
    - Muon optimizer for 2D parameters (attention and feed-forward weights)
    - AdamW optimizer for other parameters (embeddings, norms, biases)
    
    🎯 Why hybrid approach:
    - Muon works best on 2D matrices (attention, feed-forward)
    - AdamW is better for 1D parameters (embeddings, biases)
    - This gives us the best of both worlds
    
    📊 Parameter distribution:
    - Muon: ~80% of parameters (attention and feed-forward weights)
    - AdamW: ~20% of parameters (embeddings, norms, biases)
    """
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and
            'token_embedding' not in name and
            'norm' not in name and
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    # Create optimizers with different learning rates
    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)

    return [muon_optimizer, adamw_optimizer]

# =============================================================================
# 🔄 COMPONENT 9: TRAINING LOOP
# =============================================================================

def load_checkpoint(model_path: str, config: SmallModelConfig):
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

def resume_training(config: SmallModelConfig, train_loader: DataLoader, val_loader: DataLoader, checkpoint_path: str):
    """
    🔄 RESUME TRAINING FROM CHECKPOINT
    
    This function resumes training from a previously saved checkpoint.
    It loads the model state and continues training from where it left off.
    
    Args:
        config: Model configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        checkpoint_path: Path to the checkpoint file
    """
    print(f"\n🔄 RESUMING TRAINING FROM CHECKPOINT")
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    model, start_step, best_val_loss = load_checkpoint(checkpoint_path, config)
    
    # Continue with the original training function
    return _continue_training(model, config, train_loader, val_loader, start_step, best_val_loss)

def train_model(config: SmallModelConfig, train_loader: DataLoader, val_loader: DataLoader):
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
    
    🔍 Why each component matters:
    - Gradient accumulation: Allows larger effective batch sizes
    - Mixed precision: 2x faster training with minimal accuracy loss
    - Learning rate scheduling: Better convergence and final performance
    - Gradient clipping: Prevents training instability
    - Model checkpointing: Saves best model for inference
    """
    print(f"\n🚀 Training Small Qwen3 model with Muon optimizer")

    # Initialize model from scratch
    set_seed(42)
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Start training from step 0
    return _continue_training(model, config, train_loader, val_loader, start_step=0, best_val_loss=float('inf'))

def _continue_training(model, config: SmallModelConfig, train_loader: DataLoader, val_loader: DataLoader, start_step: int = 0, best_val_loss: float = float('inf')):
    """
    🔄 CONTINUE TRAINING FROM A GIVEN STATE
    
    This is the core training loop that can be used both for fresh training
    and for resuming from a checkpoint.
    
    Args:
        model: The model to train
        config: Model configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        start_step: Starting step number (0 for fresh training)
        best_val_loss: Best validation loss so far
    """
    device = next(model.parameters()).device
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")

    # Setup optimizers
    optimizers = setup_muon_optimizer(model, config)

    # Learning rate scheduling
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20  # 5% warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps  # Linear warmup
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))  # Cosine decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    # Mixed precision scaler
    scaler = GradScaler() if config.use_amp else None

    # Training loop
    model.train()
    step = start_step
    start_time = time.time()

    pbar = tqdm(total=config.max_steps, initial=start_step, desc="Training")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass with gradient accumulation
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step after accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

            # Logging
            if step % 10 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = loss.item() * config.gradient_accumulation_steps
                    perplexity = math.exp(min(current_loss, 20))

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}'
                })

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}")

                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']
                    # Save best model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': config,
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
    final_eval = evaluate_model(model, val_loader, config)
    print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, "
          f"Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'step': step,
        'final_metrics': final_eval
    }, 'models/final_model1.pt')
    print(f"💾 Saved final model to final_model1.pt")

    return model, final_eval

# =============================================================================
# 🎯 COMPONENT 10: TEXT GENERATION
# =============================================================================

def generate_text(model: nn.Module, tokenizer, prompt: str, max_length: int = 100,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9):
    """
    🔮 TEXT GENERATION FUNCTION
    
    This function generates text using the trained model with advanced sampling:
    
    🎯 Sampling Strategies:
    1. Temperature Scaling: Controls randomness (0.1 = focused, 2.0 = random)
    2. Top-k Sampling: Only consider top k most likely tokens
    3. Top-p (Nucleus) Sampling: Consider tokens until cumulative probability reaches p
    
    🔍 How it works:
    1. Tokenize the prompt
    2. For each position, get model predictions
    3. Apply temperature scaling to logits
    4. Apply top-k filtering
    5. Apply top-p filtering
    6. Sample from the filtered distribution
    7. Append to sequence and repeat
    
    📊 Parameter Effects:
    - Temperature: Lower = more focused, Higher = more creative
    - Top-k: Lower = more focused, Higher = more diverse
    - Top-p: Lower = more focused, Higher = more diverse
    """
    model.eval()
    device = next(model.parameters()).device

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(device)
    generated_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            logits = model(generated_ids)
            next_token_logits = logits[0, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits[top_k_indices] = top_k_logits

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            next_token = next_token.unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Stop if we reach the end token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    print("🏗️ Complete Qwen3 Model Ready!")
    print("\nThis file contains:")
    print("1. 🏗️ MinimalLLM: Complete language model")
    print("2. 📊 Evaluation: Model performance metrics")
    print("3. 🚀 Optimizer Setup: Hybrid Muon + AdamW")
    print("4. 🔄 Training Loop: Complete training pipeline")
    print("5. 🔮 Text Generation: Advanced sampling strategies")
    print("\nReady for training and inference!")
