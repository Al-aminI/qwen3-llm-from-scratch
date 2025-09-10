# 🎯 Universal LoRA/QLoRA Fine-tuning Guide

## 📋 Overview

The universal trainer allows you to fine-tune **ANY** model with LoRA/QLoRA, whether it's:
- **HuggingFace models** (BERT, RoBERTa, DistilBERT, etc.)
- **Your custom models** (like your MinimalLLM)
- **Any PyTorch model** with linear layers

## 🚀 Quick Start

### 1. **HuggingFace Models**

```python
from lora_qlora.core.training.universal_trainer import fine_tune_huggingface_model
from lora_qlora.core.training.config import LoRATrainingConfig

# Create config
config = LoRATrainingConfig(
    num_epochs=3,
    batch_size=8,
    learning_rate=1e-4,
    lora_rank=16,
    num_samples=1000
)

# Fine-tune any HuggingFace model
results = fine_tune_huggingface_model("bert-base-uncased", config)
results = fine_tune_huggingface_model("roberta-base", config)
results = fine_tune_huggingface_model("distilbert-base-uncased", config)
```

### 2. **Your Custom Model**

```python
from lora_qlora.core.training.universal_trainer import fine_tune_custom_model

# Fine-tune your MinimalLLM
results = fine_tune_custom_model("models/final_model1.pt", config)
```

### 3. **Manual Setup (More Control)**

```python
from lora_qlora.core.training.universal_trainer import UniversalLoRATrainer

# Initialize trainer
trainer = UniversalLoRATrainer(config)

# Setup with any model
trainer.setup_model(model_name_or_path="bert-base-uncased", model_type="huggingface")
# OR
trainer.setup_model(model_type="custom")  # Uses your MinimalLLM

# Load data and train
trainer.load_data()
results = trainer.train()
```

## 🎯 Supported Models

### **HuggingFace Models**
- ✅ **BERT**: `bert-base-uncased`, `bert-large-uncased`
- ✅ **RoBERTa**: `roberta-base`, `roberta-large`
- ✅ **DistilBERT**: `distilbert-base-uncased`
- ✅ **GPT-2**: `gpt2`, `gpt2-medium`
- ✅ **T5**: `t5-small`, `t5-base`
- ✅ **Any HuggingFace model** with `AutoModel`

### **Custom Models**
- ✅ **Your MinimalLLM**: Any model following your architecture
- ✅ **Any PyTorch model** with linear layers

## 🔧 Configuration Options

### **LoRATrainingConfig Parameters**

```python
@dataclass
class LoRATrainingConfig:
    # Model parameters
    pretrained_model_path: str = "models/final_model1.pt"  # For custom models
    tokenizer_path: str = "HuggingFaceTB/SmolLM-135M"      # Default tokenizer
    
    # LoRA parameters
    lora_rank: int = 16                    # LoRA rank (8, 16, 32, 64)
    lora_alpha: float = 32.0               # LoRA alpha (usually 2x rank)
    lora_dropout: float = 0.1              # LoRA dropout
    target_modules: List[str] = None       # Modules to apply LoRA to
    
    # Training parameters
    batch_size: int = 8                    # Batch size
    learning_rate: float = 1e-4            # Learning rate
    num_epochs: int = 3                    # Number of epochs
    max_seq_len: int = 256                 # Maximum sequence length
    
    # Data parameters
    dataset_name: str = "imdb"             # Dataset name
    num_samples: int = 1000                # Number of samples
    
    # Technical
    use_amp: bool = True                   # Mixed precision
    device: str = 'cuda' if available else 'cpu'
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    dataloader_num_workers: int = 2
```

## 📊 Examples

### **Example 1: BERT Sentiment Analysis**

```python
from lora_qlora.core.training.universal_trainer import fine_tune_huggingface_model
from lora_qlora.core.training.config import LoRATrainingConfig

# Configure for BERT
config = LoRATrainingConfig(
    num_epochs=3,
    batch_size=16,
    learning_rate=1e-4,
    lora_rank=16,
    lora_alpha=32.0,
    target_modules=["query", "key", "value", "dense"],  # BERT-specific modules
    num_samples=2000
)

# Fine-tune BERT
results = fine_tune_huggingface_model("bert-base-uncased", config)
print(f"BERT Results: {results}")
```

### **Example 2: RoBERTa with Custom Settings**

```python
# Configure for RoBERTa
config = LoRATrainingConfig(
    num_epochs=5,
    batch_size=8,
    learning_rate=2e-4,
    lora_rank=32,  # Higher rank for better performance
    lora_alpha=64.0,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # RoBERTa modules
    num_samples=5000
)

# Fine-tune RoBERTa
results = fine_tune_huggingface_model("roberta-base", config)
```

### **Example 3: Your Custom Model**

```python
# Configure for your MinimalLLM
config = LoRATrainingConfig(
    pretrained_model_path="models/final_model1.pt",
    num_epochs=3,
    batch_size=8,
    learning_rate=1e-4,
    lora_rank=16,
    target_modules=["q_proj", "k_proj", "v_proj", "w_o", "gate_proj", "up_proj", "down_proj"],
    num_samples=1000
)

# Fine-tune your model
results = fine_tune_custom_model("models/final_model1.pt", config)
```

### **Example 4: Manual Setup with Custom Dataset**

```python
from lora_qlora.core.training.universal_trainer import UniversalLoRATrainer
from lora_qlora.core.training.dataset import LoRADataset

# Create custom dataset
texts = ["This is great!", "I hate this.", "It's okay."]
labels = [1, 0, 1]  # 1=positive, 0=negative

# Initialize trainer
trainer = UniversalLoRATrainer(config)

# Setup model
trainer.setup_model(model_name_or_path="bert-base-uncased", model_type="huggingface")

# Create custom dataset
tokenizer = trainer.tokenizer
train_dataset = LoRADataset(texts, labels, tokenizer, config.max_seq_len)

# Train with custom data
trainer.train_dataset = train_dataset
results = trainer.train()
```

## 🎯 Target Modules for Different Models

### **BERT Models**
```python
target_modules = ["query", "key", "value", "dense"]
```

### **RoBERTa Models**
```python
target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
```

### **GPT-2 Models**
```python
target_modules = ["c_attn", "c_proj"]
```

### **T5 Models**
```python
target_modules = ["q", "k", "v", "o", "wi", "wo"]
```

### **Your MinimalLLM**
```python
target_modules = ["q_proj", "k_proj", "v_proj", "w_o", "gate_proj", "up_proj", "down_proj"]
```

## 🔍 How It Works

### **1. Model Detection**
The universal trainer automatically detects:
- **Model type** (HuggingFace vs Custom)
- **Hidden size** (from config or model)
- **Architecture** (transformer blocks, attention, etc.)

### **2. LoRA Application**
- **HuggingFace models**: Applies LoRA to specified modules
- **Custom models**: Uses your existing LoRA implementation
- **Automatic targeting**: Finds linear layers in specified modules

### **3. Classification Head**
- **Adaptive sizing**: Automatically determines hidden size
- **Universal interface**: Works with any model output format
- **Flexible architecture**: Can be customized for different tasks

## 🚀 Performance Tips

### **1. Choose Right LoRA Rank**
- **Small models** (BERT-base): rank = 8-16
- **Medium models** (RoBERTa-base): rank = 16-32
- **Large models** (BERT-large): rank = 32-64

### **2. Optimize Target Modules**
- **Attention layers**: Most important for performance
- **Feed-forward layers**: Good for additional capacity
- **All layers**: Maximum adaptation (more parameters)

### **3. Learning Rate Tuning**
- **HuggingFace models**: 1e-4 to 2e-4
- **Custom models**: 1e-4 to 5e-4
- **Higher rank**: Use higher learning rate

### **4. Batch Size Optimization**
- **Small models**: batch_size = 16-32
- **Medium models**: batch_size = 8-16
- **Large models**: batch_size = 4-8

## 🎉 Benefits

### **✅ Universal Compatibility**
- Works with any HuggingFace model
- Works with your custom models
- Easy to switch between models

### **✅ Minimal Changes**
- No need to modify existing code
- Same interface for all models
- Automatic configuration

### **✅ Performance**
- Maintains LoRA efficiency
- Optimized for each model type
- Best practices built-in

### **✅ Flexibility**
- Custom target modules
- Adjustable LoRA parameters
- Multiple dataset support

## 🔮 Advanced Usage

### **Custom Model Integration**

To add support for a new model type:

```python
class CustomModelClassifier(nn.Module):
    def __init__(self, base_model, num_classes=2):
        super().__init__()
        self.base_model = base_model
        # Add your custom classification head
        self.classifier = nn.Linear(base_model.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        # Your custom forward pass
        outputs = self.base_model(input_ids, attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])
```

### **Multi-task Training**

```python
# Train on multiple tasks
tasks = ["sentiment", "emotion", "topic"]
for task in tasks:
    config.dataset_name = task
    results = fine_tune_huggingface_model("bert-base-uncased", config)
    print(f"{task} results: {results}")
```

## 🏆 Success Metrics

✅ **Universal Compatibility**: Works with any model
✅ **Easy Integration**: Minimal code changes required
✅ **Performance**: Maintains LoRA efficiency
✅ **Flexibility**: Customizable for any use case
✅ **Scalability**: Handles small to large models

The universal trainer makes LoRA/QLoRA fine-tuning accessible for any model! 🚀
