# 🎯 QWEN-OMNI: Multimodal Language Model

A comprehensive package for training multimodal language models that can process both text and audio tokens using SNAC (Multi-Scale Neural Audio Codec) for state-of-the-art Text-to-Speech generation.

## ✨ Key Features

- **Multimodal Processing**: Handles both text and audio tokens seamlessly
- **SNAC Integration**: High-quality audio tokenization using Multi-Scale Neural Audio Codec
- **Qwen3 Architecture**: Modern transformer architecture with GQA and RMSNorm
- **Efficient Training**: Muon optimizer for fast convergence
- **Production Ready**: Complete training and inference pipelines
- **LibriSpeech Support**: Ready-to-use with LibriSpeech dataset

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd qwen-omni

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from qwen_omni import MultimodalPretrainingConfig, MultimodalPretrainingTrainer
from qwen_omni.pretraining.core.audio import SNACTokenizer
from qwen_omni.pretraining.core.dataset import MultimodalDataset
from transformers import AutoTokenizer

# 1. Create configuration
config = MultimodalPretrainingConfig()

# 2. Setup tokenizers
text_tokenizer = AutoTokenizer.from_pretrained('gpt2')
audio_tokenizer = SNACTokenizer("hubertsiuzdak/snac_24khz")

# 3. Create dataset
dataset = MultimodalDataset(
    data=your_audio_text_pairs,
    text_tokenizer=text_tokenizer,
    audio_tokenizer=audio_tokenizer,
    max_seq_len=config.max_seq_len
)

# 4. Train model
trainer = MultimodalPretrainingTrainer(config)
model, metrics = trainer.train(train_loader, val_loader)

# 5. Generate audio from text
audio_tokens = model.generate_audio(text_tokens, audio_tokenizer)
audio = audio_tokenizer.decode_audio(audio_tokens)
```

## 📁 Project Structure

```
qwen-omni/
├── __init__.py                 # Main package initialization
├── README.md                   # This file
├── pretraining/               # Pretraining components
│   ├── __init__.py
│   ├── core/                   # Core components
│   │   ├── audio/              # Audio processing
│   │   │   ├── __init__.py
│   │   │   └── snac_tokenizer.py
│   │   ├── dataset/            # Dataset classes
│   │   │   ├── __init__.py
│   │   │   └── multimodal_dataset.py
│   │   ├── model/              # Model components
│   │   │   ├── __init__.py
│   │   │   ├── components.py
│   │   │   └── minimal_llm.py
│   │   ├── training/           # Training components
│   │   │   ├── __init__.py
│   │   │   ├── multimodal_trainer.py
│   │   │   └── optimizer.py
│   │   └── config/            # Configuration
│   │       ├── __init__.py
│   │       └── config.py
│   ├── utils/                  # Utilities
│   │   ├── __init__.py
│   │   ├── data.py
│   │   └── generation.py
│   └── examples/               # Examples
│       ├── __init__.py
│       ├── basic/              # Basic examples
│       │   ├── multimodal_training_example.py
│       │   └── multimodal_inference_example.py
│       └── advanced/           # Advanced examples
│           └── custom_training.py
```

## 🧠 Architecture

### MultimodalLLM

The core model that can process both text and audio tokens:

- **Shared Vocabulary**: Text and audio tokens share the same embedding space
- **Qwen3 Architecture**: Pre-norm transformer with GQA and RMSNorm
- **Weight Tying**: Input and output embeddings share weights
- **Efficient Processing**: Optimized for both modalities

### SNAC Integration

- **Multi-Scale Audio**: Captures audio at multiple temporal scales
- **High Quality**: State-of-the-art audio reconstruction
- **Efficient**: Compact representation with minimal information loss
- **Flexible**: Supports various audio formats and sample rates

## 🎯 Training

### Configuration

```python
config = MultimodalPretrainingConfig(
    # Model architecture
    d_model=128,
    n_heads=4,
    n_layers=3,
    
    # Training parameters
    batch_size=8,
    max_steps=1000,
    learning_rate=1e-4,
    
    # Multimodal parameters
    max_audio_duration=10.0,
    audio_sample_rate=24000,
    audio_weight=1.0,
    text_weight=0.1
)
```

### Training Process

1. **Data Loading**: Load audio-text pairs from LibriSpeech
2. **Tokenization**: Convert text and audio to tokens
3. **Training**: Train with multimodal loss weighting
4. **Evaluation**: Monitor both text and audio performance
5. **Generation**: Test text-to-audio and audio-to-text generation

## 🎵 Audio Processing

### SNAC Tokenization

```python
# Initialize SNAC tokenizer
audio_tokenizer = SNACTokenizer("hubertsiuzdak/snac_24khz")

# Encode audio to tokens
audio_tokens = audio_tokenizer.encode_audio(audio_file)

# Decode tokens back to audio
audio = audio_tokenizer.decode_audio(audio_tokens)
```

### Audio Features

- **Sample Rate**: 24kHz (configurable)
- **Duration**: Up to 10 seconds (configurable)
- **Quality**: High-fidelity audio reconstruction
- **Efficiency**: Compact token representation

## 📊 Performance

### Model Efficiency

- **Parameters**: 7.03M (configurable)
- **Memory**: Optimized with GQA and mixed precision
- **Speed**: Fast training with Muon optimizer
- **Quality**: High-quality audio generation

### Training Metrics

- **Loss**: Cross-entropy loss for both modalities
- **Accuracy**: Token-level prediction accuracy
- **Perplexity**: Model confidence measure
- **Audio Quality**: Perceptual audio quality metrics

## 🔧 Examples

### Basic Training

```bash
cd qwen-omni
python pretraining/examples/basic/multimodal_training_example.py
```

### Inference

```bash
python pretraining/examples/basic/multimodal_inference_example.py
```

### Custom Training

```python
# Custom configuration
config = MultimodalPretrainingConfig(
    d_model=256,
    n_layers=6,
    max_steps=2000
)

# Custom dataset
dataset = MultimodalDataset(
    data=your_data,
    text_tokenizer=text_tokenizer,
    audio_tokenizer=audio_tokenizer
)

# Train
trainer = MultimodalPretrainingTrainer(config)
model, metrics = trainer.train(train_loader, val_loader)
```

## 🎯 Use Cases

### Text-to-Speech

- **Voice Synthesis**: Generate natural-sounding speech from text
- **Voice Cloning**: Adapt to different speaker characteristics
- **Multilingual**: Support for multiple languages
- **Real-time**: Fast generation for interactive applications

### Audio Understanding

- **Speech Recognition**: Convert speech to text
- **Audio Classification**: Identify audio content
- **Audio Search**: Find audio by content
- **Audio Summarization**: Generate text summaries of audio

## 🚀 Advanced Features

### Custom Audio Processing

```python
# Custom audio preprocessing
def preprocess_audio(audio_path):
    audio, sr = torchaudio.load(audio_path)
    # Custom processing here
    return audio

# Use in dataset
dataset = MultimodalDataset(
    data=data,
    text_tokenizer=text_tokenizer,
    audio_tokenizer=audio_tokenizer,
    audio_preprocessor=preprocess_audio
)
```

### Custom Loss Weighting

```python
# Custom loss weights
config.audio_weight = 2.0  # Higher weight for audio
config.text_weight = 0.5   # Lower weight for text
```

### Model Serving

```python
# Load trained model
model = MultimodalLLM(config)
model.load_state_dict(torch.load("models/best_model.pt"))

# Generate audio
audio_tokens = model.generate_audio(text_tokens, audio_tokenizer)
audio = audio_tokenizer.decode_audio(audio_tokens)
```

## 📈 Performance Tips

### Training Optimization

1. **Batch Size**: Use larger batches for better gradient estimates
2. **Learning Rate**: Start with 1e-4, adjust based on loss
3. **Gradient Accumulation**: Simulate larger batch sizes
4. **Mixed Precision**: Use AMP for faster training

### Memory Optimization

1. **GQA**: Use Grouped-Query Attention for memory efficiency
2. **Gradient Checkpointing**: Reduce memory usage during training
3. **Model Parallelism**: Distribute model across GPUs
4. **Data Parallelism**: Use multiple GPUs for training

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Audio Quality**: Check sample rate and duration settings
3. **Training Instability**: Adjust learning rate and gradient clipping
4. **Slow Training**: Use mixed precision and optimize data loading

### Debugging

```python
# Enable debugging
config.debug = True

# Check model parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Monitor training
for step, batch in enumerate(train_loader):
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
```

## 📚 API Reference

### Core Classes

#### MultimodalLLM
```python
class MultimodalLLM(nn.Module):
    def __init__(self, config)
    def forward(self, x, attention_mask=None)
    def generate_audio(self, text_tokens, audio_tokenizer, **kwargs)
    def generate_text(self, input_ids, **kwargs)
```

#### SNACTokenizer
```python
class SNACTokenizer:
    def __init__(self, model_name, device="auto")
    def encode_audio(self, audio, sample_rate=24000)
    def decode_audio(self, tokens, sample_rate=24000)
    def get_special_tokens(self)
```

#### MultimodalDataset
```python
class MultimodalDataset(Dataset):
    def __init__(self, data, text_tokenizer, audio_tokenizer, **kwargs)
    def __getitem__(self, idx)
    def get_sample_weights(self, sample)
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
- SNAC team for the audio codec
- LibriSpeech team for the dataset

## 📞 Support

For questions, issues, or contributions, please:

1. Check the existing issues
2. Create a new issue with detailed information
3. Contact the maintainers

---

**Happy Multimodal Training! 🎵🚀**
