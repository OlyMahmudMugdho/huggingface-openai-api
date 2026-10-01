# Hugging Face OpenAI-Compatible API Server

A production-ready, high-performance FastAPI server that wraps Hugging Face language models and exposes them through standard OpenAI-compatible REST endpoints (`/v1/models`, `/v1/chat/completions`, and `/v1/completions`).

Whether models are cached via `huggingface-cli`, saved in a local folder, or mounted via Kaggle Datasets (`/kaggle/input`), this server automatically discovers and serves them without requiring manual configuration or server restarts.

---

## Table of Contents

- [Overview & Architecture](#overview--architecture)
- [Key Features](#key-features)
- [Local Quickstart](#local-quickstart)
- [External Model Auto-Discovery](#external-model-auto-discovery)
- [Running in Kaggle Notebooks (Background Subprocess)](#running-in-kaggle-notebooks-background-subprocess)
- [API Reference & Usage](#api-reference--usage)
- [Configuration Reference](#configuration-reference)
- [Running Tests](#running-tests)

---

## Overview & Architecture

Many modern AI frontends and development frameworks (such as LibreChat, OpenWebUI, LangChain, LlamaIndex, and the official OpenAI SDK) expect an OpenAI-formatted API. This service acts as an abstraction layer over `transformers` and `torch`, providing:

```
┌────────────────────────────────────────────────────────┐
│  Client (OpenAI SDK / OpenWebUI / Curl / LangChain)     │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP (OpenAI API Format)
┌───────────────────────────▼────────────────────────────┐
│                  FastAPI Application                   │
│  - /v1/models             - /v1/chat/completions       │
│  - /v1/models/{model_id}  - /v1/completions (SSE)      │
└─────────────┬────────────────────────────┬─────────────┘
              │                            │
┌─────────────▼──────────────┐ ┌───────────▼─────────────┐
│    Dynamic Model Registry  │ │     Inference Engine    │
│  - HF Cache (~/.cache/hub) │ │  - Hugging Face Models  │
│  - Local Dirs (./models)   │ │  - Chat Template Engine │
│  - Kaggle (/kaggle/input)  │ │  - Streaming (SSE)      │
│  - Real-time rescanning    │ │  - LRU Memory Eviction  │
└────────────────────────────┘ └─────────────────────────┘
```

---

## Key Features

- **100% OpenAI API Compatibility**: Drop-in replacement for OpenAI API endpoints supporting streaming (SSE) and standard JSON responses.
- **Dynamic Hot-Discovery**:
  - Automatically identifies models downloaded to the Hugging Face Hub cache (`~/.cache/huggingface/hub`).
  - Scans local directories (`./models` or paths set in `LOCAL_MODELS_DIR`).
  - **Live Rescanning**: External downloads while the server is active immediately appear in `GET /v1/models` without restarting.
- **Memory & Resource Management**:
  - Auto-selects accelerator (`cuda`, `mps`, or `cpu`) and optimal precision (`bfloat16` / `float16`).
  - Configurable LRU model caching (`MAX_LOADED_MODELS`) to automatically evict idle models and avoid Out-Of-Memory (OOM) errors.
- **Chat Template Engine**: Automatically applies the model tokenizer's native Jinja chat template, with a clean fallback for raw base models.
- **Kaggle & Colab Friendly**: Seamless background execution via Python `subprocess` with health monitoring and log redirection.

---

## Local Quickstart

### 1. Prerequisites & Installation

The project uses [uv](https://github.com/astral-sh/uv) for fast, deterministic dependency management:

```bash
# Clone the repository
git clone https://github.com/your-username/huggingface-openai-api.git
cd huggingface-openai-api

# Install dependencies (FastAPI, PyTorch, Transformers, Accelerate)
uv sync
```

### 2. Start the Server

Using the packaged CLI entrypoint:
```bash
uv run huggingface-openai-api
```

Or using Uvicorn directly:
```bash
uv run uvicorn huggingface_openai_api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## External Model Auto-Discovery

You do not need to register models manually. Any model downloaded externally via either of the following mechanisms is immediately discovered:

### 1. Hugging Face CLI or `snapshot_download`
When you download a model with `huggingface-cli`:
```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```
The files land in `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct`. The server scans this cache directory and exposes it under its repo name `Qwen/Qwen2.5-0.5B-Instruct`.

### 2. Local Custom Directory (`./models` or `$LOCAL_MODELS_DIR`)
Place model files (must contain `config.json` or model weights) in `./models/`:
```text
models/
└── my-custom-model/
    ├── config.json
    ├── tokenizer.json
    └── model.safetensors
```
Or export custom directories:
```bash
export LOCAL_MODELS_DIR="/mnt/storage/models:/kaggle/input"
```
The model will be registered under `my-custom-model` and immediately served.

---

## Running in Kaggle Notebooks (Background Subprocess)

Kaggle notebooks run cell-by-cell in a single interactive session. To run the FastAPI server concurrently in the background while interacting with it from other notebook cells, launch it using Python's `subprocess.Popen`.

### Kaggle Notebook Implementation

#### Step 1: Install Dependencies
```python
# In a Kaggle Notebook Cell:
!pip install -q fastapi "uvicorn[standard]" transformers torch accelerate huggingface-hub
```

#### Step 2: Launch Server in Background
Run this cell to clone/load the server, configure search paths (including Kaggle's `/kaggle/input` datasets), and spawn the background process:

```python
import os
import subprocess
import time
import requests

# 1. Configure paths: auto-discover models in Kaggle input and HF cache
os.environ["LOCAL_MODELS_DIR"] = "/kaggle/input:./models"
os.environ["HOST"] = "127.0.0.1"
os.environ["PORT"] = "8000"
os.environ["MAX_LOADED_MODELS"] = "1"  # Preserve Kaggle GPU VRAM

# 2. Redirect output to a log file
log_file = open("server.log", "w")

# 3. Launch FastAPI server via background subprocess using run.py (handles sys.path automatically)
server_process = subprocess.Popen(
    ["python3", "run.py"],
    stdout=log_file,
    stderr=subprocess.STDOUT,
    preexec_fn=os.setsid,  # Detach process group for clean lifecycle management
)

print(f"Server launched with PID: {server_process.pid}")

# 4. Wait for server to become healthy
healthy = False
for i in range(30):
    try:
        res = requests.get("http://127.0.0.1:8000/health", timeout=1)
        if res.status_code == 200:
            healthy = True
            print("Server is healthy and ready to accept requests!")
            break
    except Exception:
        time.sleep(1)

if not healthy:
    print("Server failed to start. Last log lines:")
    with open("server.log", "r") as f:
        print(f.read())
```

#### Step 3: Inspect Logs Anytime
```python
# View recent server logs
with open("server.log", "r") as f:
    lines = f.readlines()
    print("".join(lines[-25:]))
```

#### Step 4: Interact via Official OpenAI Python Client
```python
!pip install -q openai
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="none",  # Not required unless OPENAI_API_KEY is set
)

# 1. List all available models (shows models in /kaggle/input and HF cache)
models = client.models.list()
print("Available Models:")
for m in models.data:
    print(f" - ID: {m.id} (Owner: {m.owned_by})")

# 2. Perform streaming chat completion
if models.data:
    selected_model = models.data[0].id
    print(f"\nGenerating response from: {selected_model}...")

    stream = client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": "You are a concise expert AI assistant."},
            {"role": "user", "content": "Explain gradient descent in two sentences."},
        ],
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        print(content, end="", flush=True)
    print()
```

#### Step 5: (Optional) Expose Outside Kaggle via Cloudflared Tunnel
To connect external tools (e.g., OpenWebUI or local scripts) to your Kaggle instance:
```python
# Download and start a Cloudflare tunnel
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1

tunnel_proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

# Wait for public URL
import re
for line in tunnel_proc.stdout:
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match:
        print(f"Public OpenAI Base URL: {match.group(0)}/v1")
        break
```

#### Step 6: Graceful Shutdown
When finishing your session, terminate the server background process:
```python
import signal

if server_process and server_process.poll() is None:
    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
    server_process.wait()
    print("Server stopped cleanly.")
```

---

## API Reference & Usage

### 1. List Available Models (`GET /v1/models`)

```bash
curl http://localhost:8000/v1/models
```

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-0.5B-Instruct",
      "object": "model",
      "created": 1727769600,
      "owned_by": "huggingface",
      "location": "/home/codespace/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/...",
      "permission": [...]
    }
  ]
}
```

### 2. Chat Completions (`POST /v1/chat/completions`)

#### Non-Streaming:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a python function to compute factorial."}
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }'
```

#### Streaming (Server-Sent Events):
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Tell me an interesting science fact."}
    ],
    "stream": true
  }'
```

### 3. Text Completions (`POST /v1/completions`)

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": "The future of artificial intelligence is",
    "max_tokens": 50,
    "temperature": 0.8
  }'
```

---

## Configuration Reference

Configure the server via environment variables or a `.env` file:

| Environment Variable | Default Value | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Host IP address to bind the server to |
| `PORT` | `8000` | Port to listen on |
| `OPENAI_API_KEY` | *(None)* | Optional API key. When set, requests must pass `Authorization: Bearer <key>` |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face cache root directory |
| `HF_HUB_CACHE` | `~/.cache/huggingface/hub` | Hugging Face Hub snapshot cache directory |
| `LOCAL_MODELS_DIR` | `./models` | Colon- or comma-separated list of directories to scan for local models |
| `DEVICE` | `auto` | Target device (`auto`, `cuda`, `mps`, `cpu`) |
| `TORCH_DTYPE` | `auto` | Floating point format (`auto`, `float16`, `bfloat16`, `float32`) |
| `MAX_LOADED_MODELS` | `1` | Max models kept loaded in memory before LRU eviction |
| `DEFAULT_MAX_NEW_TOKENS` | `512` | Default token generation limit if unspecified in client request |
| `TRUST_REMOTE_CODE` | `false` | Allow execution of custom code in model repositories (`true` / `false`) |

---

## Running Tests

Unit tests verify endpoint formatting, dynamic directory scanning, and streaming mechanics using mocked configurations—**without requiring heavy model weight downloads**:

```bash
uv run pytest
```
