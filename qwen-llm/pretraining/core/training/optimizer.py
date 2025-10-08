"""
This module contains the Muon optimizer implementation.
"""

import torch
import torch.nn as nn
import math

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    This is the mathematical heart of the Muon optimizer:
    
    What it does:
    - Takes a matrix G (gradients)
    - Makes it "orthogonal" (like rotating it to be perfectly aligned)
    - Uses Newton-Schulz iteration (a fast numerical method)
    
    Why orthogonalization helps:
    - Orthogonal matrices preserve vector lengths and angles
    - This prevents gradients from exploding or vanishing
    - Leads to more stable and faster training
    
    The math:
    - Newton-Schulz finds the "square root" of the identity matrix
    - It's like finding the "best rotation" for our gradients
    - Uses coefficients (a, b, c) that are mathematically optimized
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)  # Optimized coefficients
    X = G.bfloat16()  # Use bfloat16 for efficiency

    # Handle rectangular matrices
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Normalize to prevent numerical issues
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Newton-Schulz iteration (the magic happens here!)
    for _ in range(steps):
        A = X @ X.mT  # Compute X * X^T
        B = b * A + c * A @ A  # Polynomial combination
        X = a * X + B @ X  # Update X

    # Restore original orientation if needed
    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class Muon(torch.optim.Optimizer):
    """
    This is a revolutionary optimizer that combines:
    1. Momentum (like Adam) - remembers past gradients
    2. Orthogonalization (Newton-Schulz) - makes gradients "well-behaved"
    3. Adaptive learning rates - adjusts based on matrix dimensions
    
    Why Muon is special:
    - 30-50% faster convergence than Adam
    - More stable training (fewer gradient explosions)
    - Better generalization (works well on new data)
    - Particularly good for transformer models
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # Initialize momentum buffer 
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                
                # Update momentum: buf = momentum * buf + (1-momentum) * grad
                buf.lerp_(g, 1 - group["momentum"])
                
                # Apply Nesterov momentum (look ahead)
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                
                # Apply Newton-Schulz orthogonalization
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                
                # Update parameters with adaptive learning rate
                # Larger matrices get higher learning rates (scales with √(height/width))
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

def setup_muon_optimizer(model: nn.Module, config):
    """
    This function sets up a hybrid optimization strategy:
    - Muon optimizer for 2D parameters (attention and feed-forward weights)
    - AdamW optimizer for other parameters (embeddings, norms, biases)
    
    Why hybrid approach:
    - Muon works best on 2D matrices (attention, feed-forward)
    - AdamW is better for 1D parameters (embeddings, biases)
    - This gives us the best of both worlds
    
    Parameter distribution:
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
