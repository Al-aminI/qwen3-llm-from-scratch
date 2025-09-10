# 🎯 LoRA & QLoRA Fine-tuning Package

A comprehensive package for efficient fine-tuning using LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) techniques. Provides 1000x reduction in trainable parameters and 8x memory reduction for training large language models.

## ✨ Key Features

- **LoRA**: Low-rank adaptation with minimal trainable parameters
- **QLoRA**: 4-bit quantization + LoRA for maximum efficiency
- **Comprehensive quantization support**: 4-bit, 8-bit, 16-bit quantization
- **Production-ready training and serving pipelines**
- **Extensive benchmarking and evaluation tools**
- **Modular and extensible architecture**

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd lora_qlora

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from lora_qlora import LoRATrainer, QLoRATrainer, LoRATrainingConfig, QLoRATrainingConfig

# LoRA Fine-tuning
lora_config = LoRATrainingConfig(
    model_name="Qwen/Qwen2.5-0.5B",
    data_path="data/classification_data.json",
    output_dir="outputs/lora",
    num_epochs=3,
    batch_size=8,
    learning_rate=2e-4,
    lora_rank=16,
    lora_alpha=32.0
)

lora_trainer = LoRATrainer(lora_config)
lora_trainer.setup_model()
lora_trainer.load_data()
lora_trainer.setup_trainer()
lora_trainer.train()

# QLoRA Fine-tuning
qlora_config = QLoRATrainingConfig(
    model_name="Qwen/Qwen2.5-0.5B",
    data_path="data/classification_data.json",
    output_dir="outputs/qlora",
    num_epochs=3,
    batch_size=8,
    learning_rate=2e-4,
    lora_rank=16,
    lora_alpha=32.0,
    quantization_bits=4
)

qlora_trainer = QLoRATrainer(qlora_config)
qlora_trainer.setup_model()
qlora_trainer.load_data()
qlora_trainer.setup_trainer()
qlora_trainer.train()
```

## 📁 Project Structure

```
lora_qlora/
├── __init__.py                 # Main package initialization
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── core/                       # Core components
│   ├── __init__.py
│   ├── quantization/           # Quantization components
│   │   ├── __init__.py
│   │   └── quantization_expert.py
│   ├── lora/                   # LoRA components
│   │   ├── __init__.py
│   │   ├── lora_layer.py
│   │   ├── lora_linear.py
│   │   └── lora_manager.py
│   ├── qlora/                  # QLoRA components
│   │   ├── __init__.py
│   │   ├── qlora_layer.py
│   │   ├── qlora_linear.py
│   │   └── qlora_manager.py
│   └── training/               # Training components
│       ├── __init__.py
│       ├── config.py
│       ├── dataset.py
│       └── trainer.py
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── config.py              # Configuration utilities
│   ├── data.py                # Data processing utilities
│   └── serving.py             # Model serving utilities
├── examples/                   # Example scripts
│   ├── __init__.py
│   ├── basic/                 # Basic examples
│   │   ├── __init__.py
│   │   ├── lora_example.py
│   │   └── qlora_example.py
│   └── advanced/              # Advanced examples
│       ├── __init__.py
│       └── custom_training.py
└── tests/                      # Test suite
    ├── __init__.py
    ├── unit/                  # Unit tests
    │   ├── __init__.py
    │   ├── test_lora.py
    │   └── test_qlora.py
    └── integration/           # Integration tests
        ├── __init__.py
        └── test_training_pipeline.py
```

## 🔧 Core Components

### LoRA Components

- **LoRALayer**: Core LoRA layer implementation
- **LoRALinear**: LoRA-adapted linear layer
- **LoRAManager**: Manages LoRA application to entire models

### QLoRA Components

- **QLoRALayer**: QLoRA layer with 4-bit quantization
- **QLoRALinear**: QLoRA-adapted linear layer
- **QLoRAManager**: Manages QLoRA application to entire models

### Training Components

- **LoRATrainingConfig**: Configuration for LoRA training
- **QLoRATrainingConfig**: Configuration for QLoRA training
- **LoRADataset**: Dataset class for LoRA training
- **QLoRADataset**: Dataset class for QLoRA training
- **LoRATrainer**: Trainer for LoRA fine-tuning
- **QLoRATrainer**: Trainer for QLoRA fine-tuning

## 🛠️ Utilities

### Configuration Utilities

```python
from lora_qlora.utils.config import load_config, save_config, merge_configs

# Load configuration
config = load_config("config.yaml")

# Save configuration
save_config(config, "output_config.json")

# Merge configurations
merged_config = merge_configs(base_config, override_config)
```

### Data Processing Utilities

```python
from lora_qlora.utils.data import load_data, preprocess_data, split_data

# Load and process data
data = load_data("data.json")
processed_data = preprocess_data(data)
train_data, val_data, test_data = split_data(processed_data)
```

### Model Serving Utilities

```python
from lora_qlora.utils.serving import ModelServer, InferenceEngine

# Create model server
server = ModelServer("path/to/model", model_type="lora")
server.load_model()

# Create inference engine
engine = InferenceEngine(server)

# Make predictions
result = engine.predict("Your text here")
```

## 📊 Examples

### Basic Examples

Run the basic examples to get started:

```bash
# LoRA example
python examples/basic/lora_example.py

# QLoRA example
python examples/basic/qlora_example.py
```

### Advanced Examples

Explore advanced features:

```bash
# Custom training with hyperparameter search
python examples/advanced/custom_training.py
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run unit tests only
python -m pytest tests/unit/

# Run integration tests only
python -m pytest tests/integration/
```

## 📈 Performance Benefits

### LoRA Benefits

- **1000x reduction** in trainable parameters
- **Faster training** with minimal accuracy loss
- **Memory efficient** fine-tuning
- **Easy deployment** with small adapter weights

### QLoRA Benefits

- **8x memory reduction** compared to full fine-tuning
- **4-bit quantization** for maximum efficiency
- **Maintains performance** with significant memory savings
- **Ideal for resource-constrained environments**

## 🔧 Configuration

### LoRA Configuration

```python
lora_config = LoRATrainingConfig(
    model_name="Qwen/Qwen2.5-0.5B",
    data_path="data/classification_data.json",
    output_dir="outputs/lora",
    num_epochs=3,
    batch_size=8,
    learning_rate=2e-4,
    lora_rank=16,           # LoRA rank
    lora_alpha=32.0,        # LoRA alpha scaling
    lora_dropout=0.1,       # LoRA dropout
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

### QLoRA Configuration

```python
qlora_config = QLoRATrainingConfig(
    model_name="Qwen/Qwen2.5-0.5B",
    data_path="data/classification_data.json",
    output_dir="outputs/qlora",
    num_epochs=3,
    batch_size=8,
    learning_rate=2e-4,
    lora_rank=16,
    lora_alpha=32.0,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    quantization_bits=4     # 4-bit quantization
)
```

## 📚 API Reference

### Core Classes

#### LoRALayer
```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=1.0, dropout=0.0)
    def forward(self, x)
    def get_parameter_count(self)
    def reset_parameters(self)
```

#### QLoRALayer
```python
class QLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=1.0, dropout=0.0)
    def forward(self, x)
    def quantize_weights(self, weights)
    def dequantize_weights(self, quantized_weights)
    def get_memory_usage(self)
```

#### LoRAManager
```python
class LoRAManager:
    def __init__(self, model, config)
    def apply_lora(self, target_modules=None)
    def get_trainable_parameters(self)
    def get_parameter_count(self)
    def get_memory_usage(self)
    def save_lora_weights(self, path)
    def load_lora_weights(self, path)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers for the base model implementations
- Qwen team for the Qwen models
- LoRA and QLoRA research papers for the theoretical foundation

## 📞 Support

For questions, issues, or contributions, please:

1. Check the existing issues
2. Create a new issue with detailed information
3. Contact the maintainers

---

**Happy Fine-tuning! 🚀**
