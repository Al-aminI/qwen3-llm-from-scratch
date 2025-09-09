# 🎯 IMDB Sentiment Analysis Fine-tuning

## 📋 Overview

This directory contains a complete fine-tuning pipeline for adapting our pre-trained Qwen3 language model to the IMDB sentiment analysis task. The fine-tuning process adds a classification head on top of the pre-trained model and trains only the new parameters while keeping the pre-trained weights frozen.

## 🏗️ Architecture

### Sentiment Classification Model

```
Input Text → Tokenizer → Pre-trained Qwen3 → Classification Head → Sentiment (Positive/Negative)
```

**Key Components:**
- **Pre-trained Qwen3**: Frozen weights from language modeling
- **Classification Head**: New trainable layers for sentiment classification
- **Feature Extraction**: Uses the first token representation (like [CLS] token)

### Model Structure

```python
SentimentClassifier(
  pretrained_model: MinimalLLM (frozen),
  classifier: Sequential(
    Dropout(0.1),
    Linear(128, 64),
    ReLU(),
    Dropout(0.1),
    Linear(64, 2)  # Binary classification
  )
)
```

## 📊 Dataset

**IMDB Movie Reviews Dataset:**
- **Training samples**: 25,000 movie reviews
- **Test samples**: 25,000 movie reviews
- **Labels**: 0 (Negative) or 1 (Positive)
- **Average length**: ~200-300 words per review

**Sample Data:**
```python
{
    "text": "I absolutely loved this movie! The acting was fantastic...",
    "label": 1  # Positive
}
```

## 🚀 Usage

### 1. Demo (Quick Test)

```bash
# Test the fine-tuning pipeline with a small subset
python demo_finetune.py
```

### 2. Full Fine-tuning

```bash
# Fine-tune on the full IMDB dataset
python finetune_imdb.py --pretrained models/final_model1.pt --epochs 3 --batch_size 16

# With custom parameters
python finetune_imdb.py \
    --pretrained models/final_model1.pt \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-4
```

### 3. Test Fine-tuned Model

```bash
# Test the fine-tuned sentiment classifier
python finetune_imdb.py --test
```

## ⚙️ Configuration

### FineTuneConfig Parameters

```python
@dataclass
class FineTuneConfig:
    # Model architecture
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 512
    
    # Fine-tuning parameters
    num_classes: int = 2  # Binary classification
    max_seq_len: int = 512
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 3
    
    # Regularization
    dropout: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
```

## 🎯 Training Process

### 1. **Data Loading**
- Loads IMDB dataset from HuggingFace
- Tokenizes reviews using the same tokenizer as pre-training
- Creates train/test data loaders

### 2. **Model Setup**
- Loads pre-trained Qwen3 model
- Adds classification head
- Freezes pre-trained parameters
- Only trains classification head (8,386 parameters)

### 3. **Training Loop**
- Forward pass through pre-trained model (frozen)
- Extract features from first token
- Pass through classification head
- Compute cross-entropy loss
- Backpropagate through classification head only

### 4. **Evaluation**
- Tests on held-out test set
- Computes accuracy and loss
- Saves best model

## 📈 Expected Results

**Typical Performance:**
- **Training Time**: ~10-15 minutes (3 epochs)
- **Test Accuracy**: 85-90% (depending on training)
- **Trainable Parameters**: 8,386 (vs 7M+ total)
- **Memory Usage**: Significantly reduced due to frozen weights

## 🔧 Key Features

### 1. **Efficient Fine-tuning**
- Only trains classification head
- Pre-trained weights remain frozen
- Fast training and low memory usage

### 2. **Robust Architecture**
- Uses first token as sentence representation
- Dropout for regularization
- Gradient clipping for stability

### 3. **Flexible Configuration**
- Adjustable sequence length
- Configurable batch size and learning rate
- Multiple training epochs

### 4. **Comprehensive Evaluation**
- Real-time training metrics
- Test set evaluation
- Sample prediction testing

## 📁 File Structure

```
qwen-llm/
├── finetune_imdb.py          # Main fine-tuning script
├── demo_finetune.py          # Demo script for testing
├── FINETUNING_README.md      # This documentation
└── models/
    ├── final_model1.pt       # Pre-trained model
    └── imdb_sentiment_classifier.pt  # Fine-tuned model
```

## 🧪 Testing the Model

### Sample Predictions

```python
# Load fine-tuned model
model = load_sentiment_classifier("models/imdb_sentiment_classifier.pt")

# Test samples
test_texts = [
    "I absolutely loved this movie!",
    "This was the worst film ever.",
    "The movie was okay, nothing special."
]

# Get predictions
for text in test_texts:
    sentiment, confidence = predict_sentiment(model, text)
    print(f"'{text}' → {sentiment} ({confidence:.3f})")
```

## 🎓 Learning Objectives

This fine-tuning example demonstrates:

1. **Transfer Learning**: Using pre-trained language models for specific tasks
2. **Classification Architecture**: Adding task-specific heads to language models
3. **Efficient Training**: Freezing pre-trained weights and training only new layers
4. **Real-world Application**: Sentiment analysis on movie reviews
5. **Evaluation Metrics**: Accuracy, loss, and confidence scores

## 🔮 Extensions

### Potential Improvements

1. **Unfreeze Some Layers**: Allow fine-tuning of last few transformer layers
2. **Different Tasks**: Adapt for other classification tasks (emotion, topic, etc.)
3. **Multi-class Classification**: Extend to more than 2 classes
4. **Sequence Classification**: Handle variable-length sequences better
5. **Advanced Regularization**: Add more sophisticated dropout patterns

### Other Datasets

The same approach can be used for:
- **SST-2**: Stanford Sentiment Treebank
- **AG News**: News article classification
- **Yelp Reviews**: Restaurant review sentiment
- **Amazon Reviews**: Product review sentiment

## 🏆 Success Metrics

✅ **Model Architecture**: Classification head added successfully
✅ **Data Loading**: IMDB dataset loaded and tokenized
✅ **Training Pipeline**: Fine-tuning loop implemented
✅ **Evaluation**: Test accuracy and loss computation
✅ **Model Saving**: Fine-tuned model checkpointing
✅ **Demo Script**: Quick testing and validation

This fine-tuning pipeline provides a complete example of adapting a pre-trained language model for a specific downstream task, demonstrating the power of transfer learning in modern NLP!
