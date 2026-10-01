import json
import shutil
import tempfile
from pathlib import Path
import pytest

from huggingface_openai_api.config import settings
from huggingface_openai_api.model_registry import ModelRegistry


@pytest.fixture
def temp_dirs():
    """Create temporary directories for HF cache and local models."""
    temp_dir = Path(tempfile.mkdtemp())
    hf_cache = temp_dir / "hf_hub"
    local_dir = temp_dir / "models"
    hf_cache.mkdir(parents=True)
    local_dir.mkdir(parents=True)

    # Backup original settings
    orig_hf_cache = settings.hf_hub_cache
    orig_local_dirs = settings.local_models_dirs

    settings.hf_hub_cache = hf_cache
    settings.local_models_dirs = [local_dir]

    yield hf_cache, local_dir

    # Restore settings and cleanup
    settings.hf_hub_cache = orig_hf_cache
    settings.local_models_dirs = orig_local_dirs
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_empty_registry(temp_dirs):
    registry = ModelRegistry()
    models = registry.list_models()
    assert models == []


def test_discover_local_downloaded_model(temp_dirs):
    hf_cache, local_dir = temp_dirs

    # Simulate an externally downloaded model in local_dir
    model_dir = local_dir / "my-custom-llm"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (model_dir / "tokenizer_config.json").write_text(json.dumps({}))

    registry = ModelRegistry()
    models = registry.list_models()

    assert len(models) == 1
    assert models[0].id == "my-custom-llm"
    assert models[0].owned_by == "local"
    assert models[0].location == str(model_dir)

    # Test retrieval
    found = registry.get_model("my-custom-llm")
    assert found is not None
    assert found.id == "my-custom-llm"

    # Test path resolution
    resolved = registry.resolve_model_path("my-custom-llm")
    assert resolved == str(model_dir)


def test_discover_hf_cache_model(temp_dirs):
    hf_cache, local_dir = temp_dirs

    # Simulate a model downloaded via huggingface-cli / snapshot_download
    repo_dir = hf_cache / "models--meta-llama--Llama-3.2-1B"
    snapshots_dir = repo_dir / "snapshots" / "abc123def456"
    snapshots_dir.mkdir(parents=True)
    (snapshots_dir / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))

    registry = ModelRegistry()
    models = registry.list_models()

    assert len(models) == 1
    assert models[0].id == "meta-llama/Llama-3.2-1B"
    assert models[0].owned_by == "huggingface"
    assert models[0].location == str(snapshots_dir)

    # Resolution by full ID
    assert registry.resolve_model_path("meta-llama/Llama-3.2-1B") == str(snapshots_dir)
    # Resolution by short name
    assert registry.resolve_model_path("Llama-3.2-1B") == str(snapshots_dir)


def test_dynamic_model_discovery(temp_dirs):
    hf_cache, local_dir = temp_dirs
    registry = ModelRegistry()

    assert len(registry.list_models()) == 0

    # User downloads a model while app is running
    new_model = local_dir / "downloaded-later"
    new_model.mkdir()
    (new_model / "config.json").write_text(json.dumps({}))

    # Rescanning should immediately show the new model
    models = registry.list_models()
    assert len(models) == 1
    assert models[0].id == "downloaded-later"
