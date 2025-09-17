# 🚀 OpenAI-Compatible API Server

A production-ready OpenAI-compatible API server powered by the fast_inference engine. This implementation provides full compatibility with the OpenAI API specification while leveraging the high-performance inference capabilities of our engine.

## ✨ Features

- **🔌 Full OpenAI Compatibility**: Complete implementation of OpenAI API endpoints
- **⚡ High Performance**: Powered by fast_inference with KV caching and PagedAttention
- **🌐 Universal Model Support**: Works with any model (MinimalLLM, HuggingFace, custom)
- **📡 Streaming Support**: Real-time response streaming with Server-Sent Events
- **🛡️ Production Ready**: Comprehensive error handling, logging, and monitoring
- **🔧 Easy Integration**: Simple deployment and configuration

## 📊 Supported Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions with streaming support |
| `/v1/completions` | POST | Text completions with streaming support |
| `/v1/embeddings` | POST | Text embeddings generation |
| `/v1/models` | GET | List available models |
| `/v1/tokenize` | POST | Tokenize text |
| `/v1/detokenize` | POST | Detokenize tokens |
| `/health` | GET | Health check endpoint |
| `/ping` | GET | Simple ping endpoint |

## 🚀 Quick Start

### 1. Start the Server

```bash
# Using the CLI
python -m fast_inference.cli_openai \
    --model-path models/final_model1.pt \
    --tokenizer-path HuggingFaceTB/SmolLM-135M \
    --host 0.0.0.0 \
    --port 8000

# Or using environment variables
export MODEL_PATH="models/final_model1.pt"
export TOKENIZER_PATH="HuggingFaceTB/SmolLM-135M"
export MODEL_NAME="my-custom-model"

python -m fast_inference.cli_openai --host 0.0.0.0 --port 8000
```

### 2. Test the API

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{
    "model": "fast-inference-model",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### 3. Use with OpenAI Client

```python
import openai

# Configure client
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Use any string as API key
)

# Chat completion
response = client.chat.completions.create(
    model="fast-inference-model",
    messages=[
        {"role": "user", "content": "Tell me a joke"}
    ],
    max_tokens=100,
    temperature=0.8
)

print(response.choices[0].message.content)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to model file | `models/final_model1.pt` |
| `TOKENIZER_PATH` | Path to tokenizer | `HuggingFaceTB/SmolLM-135M` |
| `MODEL_NAME` | Model name for API | `fast-inference-model` |

### CLI Arguments

```bash
python -m fast_inference.cli_openai --help
```

```
usage: cli_openai.py [-h] [--host HOST] [--port PORT] [--model-path MODEL_PATH]
                     [--tokenizer-path TOKENIZER_PATH] [--model-name MODEL_NAME]
                     [--workers WORKERS] [--log-level LOG_LEVEL]

Fast Inference OpenAI-Compatible API Server

options:
  -h, --help            show this help message and exit
  --host HOST           Host to bind to (default: 0.0.0.0)
  --port PORT           Port to bind to (default: 8000)
  --model-path MODEL_PATH
                        Path to model file
  --tokenizer-path TOKENIZER_PATH
                        Path to tokenizer
  --model-name MODEL_NAME
                        Model name (default: fast-inference-model)
  --workers WORKERS     Number of worker processes (default: 1)
  --log-level LOG_LEVEL
                        Log level (default: info)
```

## 📝 API Examples

### Chat Completions

```python
# Non-streaming
response = client.chat.completions.create(
    model="fast-inference-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=50,
    temperature=0.7
)

# Streaming
stream = client.chat.completions.create(
    model="fast-inference-model",
    messages=[
        {"role": "user", "content": "Tell me a story"}
    ],
    max_tokens=200,
    temperature=0.8,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Text Completions

```python
# Non-streaming
response = client.completions.create(
    model="fast-inference-model",
    prompt="The future of AI is",
    max_tokens=50,
    temperature=0.8
)

# Streaming
stream = client.completions.create(
    model="fast-inference-model",
    prompt="Once upon a time",
    max_tokens=100,
    temperature=0.9,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].text:
        print(chunk.choices[0].text, end="")
```

### Embeddings

```python
response = client.embeddings.create(
    model="fast-inference-model",
    input="This is a test sentence for embedding generation."
)

embedding = response.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")
```

## 🏗️ Architecture

### Components

1. **OpenAI Protocol Models** (`openai_protocol.py`)
   - Request/response schemas following OpenAI specification
   - Comprehensive validation and error handling

2. **Serving Engine** (`openai_serving_engine.py`)
   - Core logic for processing requests
   - Integration with fast_inference engines
   - Streaming and non-streaming response handling

3. **API Server** (`openai_api_server.py`)
   - FastAPI-based HTTP server
   - Route definitions and middleware
   - Health checks and monitoring

4. **CLI Interface** (`cli_openai.py`)
   - Command-line interface for server management
   - Configuration and deployment options

### Request Flow

```
Client Request → FastAPI → Serving Engine → Universal Engine → Model
                     ↓
Client Response ← JSON/Stream ← Response Formatter ← Generated Text
```

## 🔍 Monitoring and Debugging

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Ping endpoint
curl http://localhost:8000/ping
```

### Logging

The server provides comprehensive logging:

```bash
# Set log level
python -m fast_inference.cli_openai --log-level debug
```

### Error Handling

The API provides standardized error responses:

```json
{
  "error": {
    "message": "Validation error: field required",
    "type": "invalid_request_error",
    "param": "messages",
    "code": 400
  }
}
```

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -e .

EXPOSE 8000

CMD ["python", "-m", "fast_inference.cli_openai", "--host", "0.0.0.0", "--port", "8000"]
```

### Load Balancing

```yaml
# docker-compose.yml
version: '3.8'
services:
  api-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/model.pt
      - TOKENIZER_PATH=HuggingFaceTB/SmolLM-135M
    deploy:
      replicas: 3
```

### Reverse Proxy (Nginx)

```nginx
upstream fast_inference {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://fast_inference;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🧪 Testing

### Run Test Suite

```bash
# Start the server in one terminal
python -m fast_inference.cli_openai

# Run tests in another terminal
python examples/openai_api_example.py
```

### Manual Testing

```bash
# Test chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fast-inference-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 10
  }'

# Test streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fast-inference-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 10,
    "stream": true
  }'
```

## 🔧 Customization

### Custom Model Integration

```python
from fast_inference.core.engine import create_universal_vllm_style_engine

# Create custom engine
engine = create_universal_vllm_style_engine(
    model="path/to/your/model.pt",
    tokenizer="path/to/your/tokenizer",
    max_seq_len=4096,
    num_blocks=2000,
    block_size=256
)

# Use with API server
from fast_inference.core.engine import OpenAIServingEngine

serving_engine = OpenAIServingEngine(engine, "my-custom-model")
```

### Custom Endpoints

```python
from fastapi import FastAPI
from fast_inference.core.engine.openai_api_server import app

# Add custom endpoints
@app.post("/v1/custom-endpoint")
async def custom_endpoint():
    return {"message": "Custom endpoint"}

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Happy coding! 🚀**

For more information, visit our [main documentation](README.md) or check out the [examples](examples/).
