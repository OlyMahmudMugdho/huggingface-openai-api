import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from huggingface_openai_api.app import app
from huggingface_openai_api.config import settings
from huggingface_openai_api.schemas import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionChoice,
    CompletionResponse,
    Usage,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def temp_models_dir():
    temp_dir = Path(tempfile.mkdtemp())
    local_dir = temp_dir / "models"
    local_dir.mkdir()

    orig_dirs = settings.local_models_dirs
    settings.local_models_dirs = [local_dir]

    yield local_dir

    settings.local_models_dirs = orig_dirs
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_health_and_root(client):
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "running"

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"


def test_list_models_empty_and_populated(client, temp_models_dir):
    # Initially empty
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    initial_count = len(data["data"])

    # Add external model
    model_dir = temp_models_dir / "gpt2-local"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({}))

    # Rescan via endpoint
    res2 = client.get("/v1/models")
    assert res2.status_code == 200
    data2 = res2.json()
    assert len(data2["data"]) == initial_count + 1
    ids = [m["id"] for m in data2["data"]]
    assert "gpt2-local" in ids


def test_get_single_model(client, temp_models_dir):
    model_dir = temp_models_dir / "my-test-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({}))

    # Exists
    res = client.get("/v1/models/my-test-model")
    assert res.status_code == 200
    assert res.json()["id"] == "my-test-model"

    # Not found
    res_404 = client.get("/v1/models/non-existent-model")
    assert res_404.status_code == 404
    error_data = res_404.json()
    assert "error" in error_data
    assert error_data["error"]["code"] == "model_not_found"


def test_chat_completion_non_streaming(client):
    mock_resp = ChatCompletionResponse(
        id="chatcmpl-test-123",
        model="my-test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="Hello! How can I help you today?"),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
    )

    with patch("huggingface_openai_api.routes.chat.inference_engine.generate_chat", return_value=mock_resp):
        payload = {
            "model": "my-test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.7,
        }
        res = client.post("/v1/chat/completions", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == "chatcmpl-test-123"
        assert data["choices"][0]["message"]["content"] == "Hello! How can I help you today?"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["usage"]["total_tokens"] == 18


def test_chat_completion_streaming(client):
    async def mock_stream_gen(req):
        chunks = [
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"my-model","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"my-model","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"my-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ]
        for c in chunks:
            yield c

    with patch("huggingface_openai_api.routes.chat.inference_engine.generate_chat_stream", side_effect=mock_stream_gen):
        payload = {
            "model": "my-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        res = client.post("/v1/chat/completions", json=payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        body = res.text
        assert "data: [DONE]" in body
        assert '"content":"Hi"' in body


def test_completion_non_streaming(client):
    mock_resp = CompletionResponse(
        id="cmpl-test-123",
        model="my-test-model",
        choices=[
            CompletionChoice(
                text=" world!",
                index=0,
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=2, completion_tokens=2, total_tokens=4),
    )

    with patch("huggingface_openai_api.routes.completions.inference_engine.generate_completion", return_value=mock_resp):
        payload = {
            "model": "my-test-model",
            "prompt": "Hello",
        }
        res = client.post("/v1/completions", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["choices"][0]["text"] == " world!"
        assert data["usage"]["total_tokens"] == 4


def test_api_key_auth(client):
    orig_key = settings.api_key
    settings.api_key = "secret-token-123"

    try:
        # Without header -> 401
        res_no_auth = client.get("/v1/models")
        assert res_no_auth.status_code == 401
        assert res_no_auth.json()["error"]["code"] == "invalid_api_key"

        # With wrong header -> 401
        res_wrong_auth = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
        assert res_wrong_auth.status_code == 401

        # With valid header -> 200
        res_valid = client.get("/v1/models", headers={"Authorization": "Bearer secret-token-123"})
        assert res_valid.status_code == 200
    finally:
        settings.api_key = orig_key
