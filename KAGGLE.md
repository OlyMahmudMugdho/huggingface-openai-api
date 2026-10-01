# Running Hugging Face OpenAI API in Kaggle Notebooks

This guide provides ready-to-run code cells to serve Hugging Face models using the OpenAI API standard inside a Kaggle notebook environment.

---

## 1. Quick Setup & Installation

Run this in your first Kaggle notebook cell:

```python
# Install requirements
!pip install -q fastapi "uvicorn[standard]" transformers torch accelerate huggingface-hub openai requests
```

---

## 2. Launch Server in Background using Python `subprocess`

Run this cell to start the server in a non-blocking background process and verify health:

```python
import os
import signal
import subprocess
import time
import requests

# Set environment configuration
os.environ["LOCAL_MODELS_DIR"] = "/kaggle/input:./models"
os.environ["HOST"] = "127.0.0.1"
os.environ["PORT"] = "8000"
os.environ["MAX_LOADED_MODELS"] = "1"  # Keep memory footprint lean on Kaggle GPUs

# Open log file to capture server stdout/stderr
server_log = open("server.log", "w")

# Start server as a background subprocess using run.py (automatically configures src path)
server_process = subprocess.Popen(
    ["python3", "run.py"],
    stdout=server_log,
    stderr=subprocess.STDOUT,
    preexec_fn=os.setsid,  # Clean process group management
)

print(f"Server launched in background with PID: {server_process.pid}")

# Wait for server health endpoint to respond
healthy = False
for attempt in range(30):
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

---

## 3. Serving Models in Kaggle

You have three convenient ways to serve models in Kaggle:

### Option A: Use Models Pre-Attached from Kaggle Datasets / Models
If you attach a model via Kaggle's "+ Add Input" (e.g. `Qwen2.5-0.5B-Instruct`), Kaggle mounts it under `/kaggle/input/<dataset-name>/`.
Because `LOCAL_MODELS_DIR="/kaggle/input"`, the server **automatically discovers** and lists it in `/v1/models` without downloading!

### Option B: Download with `huggingface-cli`
```python
!huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```
The model files are saved directly to Hugging Face Hub cache (`~/.cache/huggingface/hub/`) and immediately become visible in `/v1/models`.

### Option C: Download with `snapshot_download`
```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="Qwen/Qwen2.5-0.5B-Instruct")
```

---

## 4. Querying the Server (OpenAI SDK)

Use standard OpenAI client code:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="none",
)

# 1. List all detected models
model_list = client.models.list()
print("Discovered models:")
for m in model_list.data:
    print(f" - {m.id} (Source: {m.owned_by})")

# 2. Chat completion with streaming
if model_list.data:
    target_model = model_list.data[0].id
    print(f"\nQuerying model: {target_model}\n" + "-"*40)

    stream = client.chat.completions.create(
        model=target_model,
        messages=[
            {"role": "system", "content": "You are a helpful and concise assistant."},
            {"role": "user", "content": "What are three advantages of FastAPI?"},
        ],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
    print()
```

---

## 5. Checking Server Logs in Notebook

If you need to diagnose or inspect requests:

```python
with open("server.log", "r") as f:
    logs = f.readlines()
    print("".join(logs[-30:]))
```

---

## 6. Exposing Outside Kaggle (Optional Public URL)

To connect an external frontend (e.g. OpenWebUI or local machine) to your Kaggle server:

```python
import re
import subprocess

# Install cloudflared
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1

# Run tunnel
tunnel = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

for line in tunnel.stdout:
    url_match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if url_match:
        print("Your public OpenAI API Base URL is:")
        print(f"{url_match.group(0)}/v1")
        break
```

---

## 7. Graceful Shutdown

When your notebook work is done:

```python
import os
import signal

if "server_process" in locals() and server_process.poll() is None:
    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
    server_process.wait()
    print("Server stopped.")
```
