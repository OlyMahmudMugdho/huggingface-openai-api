# Hugging Face OpenAI-Compatible API Server

A lightweight, high-performance FastAPI server that exposes Hugging Face models through the standard OpenAI API format (`/v1/models`, `/v1/chat/completions`, and `/v1/completions`).

## Key Features

- **OpenAI Compatible**: Seamlessly works with the official `openai` Python/Node SDKs, OpenWebUI, LibreChat, and standard HTTP clients.
- **External Model Auto-Discovery**:
  - Automatically scans and lists models downloaded via Hugging Face Hub (`huggingface-cli download`, `snapshot_download`, etc.) from `~/.cache/huggingface/hub` or `$HF_HUB_CACHE`.
  - Automatically discovers local models placed in `./models/` or paths configured via `LOCAL_MODELS_DIR` / `MODELS_DIR`.
  - **Dynamic Rescanning**: Models downloaded externally while the server is running immediately appear in `GET /v1/models` without server restarts.
- **Streaming Support**: Full Server-Sent Events (SSE) streaming support for `/v1/chat/completions` and `/v1/completions`.
- **Chat Template Support**: Automatically utilizes the model's tokenizer chat template (with fallback formatting if absent).
- **Resource Management**: Device auto-detection (`cuda`, `mps`, `cpu`), automatic float16/bfloat16 precision, and LRU model memory management to avoid Out-Of-Memory (OOM).

---

## Installation & Setup

This project is managed with [uv](https://github.com/astral-sh/uv).

```bash
# Clone the repository
git clone <repo-url>
cd huggingface-openai-api

# Install dependencies (FastAPI, PyTorch, Transformers, Accelerate)
uv sync
```

---

## Running the Server

Start the API server using uv:

```bash
uv run huggingface-openai-api
```

Or run via uvicorn directly:

```bash
uv run uvicorn huggingface_openai_api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## How External Model Discovery Works

You can download models externally using either of the following approaches:

### Method 1: Using Hugging Face CLI or Cache
If you download a model using `huggingface-cli` or `snapshot_download`:
```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```
The model files are saved to `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct`. The server dynamically inspects this directory:
- It appears in `GET /v1/models` as `Qwen/Qwen2.5-0.5B-Instruct`.
- You can query it in `/v1/chat/completions` using `"model": "Qwen/Qwen2.5-0.5B-Instruct"` or `"Qwen2.5-0.5B-Instruct"`.

### Method 2: Local Directory (`./models` or `LOCAL_MODELS_DIR`)
Download or save model weights (containing `config.json` and weight files) into `./models/`:
```bash
mkdir -p models/my-custom-llm
# Place config.json, tokenizer.json, model.safetensors into models/my-custom-llm/
```
Or point to any custom directory on your system:
```bash
export LOCAL_MODELS_DIR="/path/to/my/models:/another/path/to/models"
```
The server immediately detects any subdirectories containing valid model files and lists them in `GET /v1/models`.

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8000` | Bind port |
| `OPENAI_API_KEY` | *(None)* | Optional API key to secure endpoints with Bearer auth |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face home directory |
| `HF_HUB_CACHE` | `~/.cache/huggingface/hub` | Hugging Face Hub cache directory |
| `LOCAL_MODELS_DIR` | `./models` | Colon- or comma-separated list of local model folders |
| `DEVICE` | `auto` | Device to run inference on (`auto`, `cuda`, `mps`, `cpu`) |
| `TORCH_DTYPE` | `auto` | Torch data type (`auto`, `float16`, `bfloat16`, `float32`) |
| `MAX_LOADED_MODELS` | `1` | Maximum number of models kept concurrently in memory |
| `DEFAULT_MAX_NEW_TOKENS` | `512` | Default max generation tokens if unspecified |

---

## Usage Examples

### 1. List Available Models
```bash
curl http://localhost:8000/v1/models
```
Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-0.5B-Instruct",
      "object": "model",
      "created": 1727769600,
      "owned_by": "huggingface",
      "location": "/home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/..."
    }
  ]
}
```

### 2. Chat Completions (Non-Streaming)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain recursion in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### 3. Chat Completions (Streaming)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Write a short poem about coding."}
    ],
    "stream": true
  }'
```

### 4. Using the Official OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",  # not needed unless OPENAI_API_KEY is configured
)

# List models
models = client.models.list()
for model in models:
    print(f"Discovered model: {model.id}")

# Chat completion
response = client.chat.completions.create(
    model=models.data[0].id,
    messages=[
        {"role": "user", "content": "What is FastAPI?"}
    ],
    stream=True,
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
print()
```

---

## Running Tests

Tests verify the registry and API endpoints using mock and directory scanning without needing to download large model weights:

```bash
uv run pytest
```
