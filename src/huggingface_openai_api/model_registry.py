import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from .config import settings
from .schemas import ModelObject, ModelPermission

logger = logging.getLogger(__name__)


class ModelInfo:
    def __init__(
        self,
        model_id: str,
        path: Path,
        source: str,  # "hf_cache" or "local"
        created: int = 0,
        size_bytes: Optional[int] = None,
    ):
        self.model_id = model_id
        self.path = path
        self.source = source
        self.created = created
        self.size_bytes = size_bytes

    def to_model_object(self) -> ModelObject:
        return ModelObject(
            id=self.model_id,
            object="model",
            created=self.created or int(self.path.stat().st_mtime if self.path.exists() else 0),
            owned_by=self.source,
            permission=[ModelPermission()],
            location=str(self.path),
            size_bytes=self.size_bytes,
        )


class ModelRegistry:
    """
    Dynamically scans and discovers Hugging Face models from:
    1. Hugging Face Hub cache directory (~/.cache/huggingface/hub or $HF_HUB_CACHE)
    2. External / local model directories (e.g. ./models or $LOCAL_MODELS_DIR)
    3. Direct local paths
    """

    def __init__(self):
        self._cached_models: Dict[str, ModelInfo] = {}

    def scan_hf_cache(self) -> Dict[str, ModelInfo]:
        """Scan Hugging Face Hub cache directory for downloaded models."""
        models: Dict[str, ModelInfo] = {}
        cache_dir = settings.hf_hub_cache

        # Try using huggingface_hub scan_cache_dir if available
        try:
            from huggingface_hub import scan_cache_dir

            if cache_dir.exists():
                hf_cache = scan_cache_dir(cache_dir=str(cache_dir))
                for repo in hf_cache.repos:
                    if repo.repo_type == "model":
                        # Find the main snapshot or latest snapshot path
                        snapshot_path = None
                        if repo.revisions:
                            # Revisions sorted or take latest
                            latest_rev = max(repo.revisions, key=lambda r: r.last_modified)
                            snapshot_path = latest_rev.snapshot_path

                        model_path = Path(snapshot_path) if snapshot_path else Path(repo.repo_path)
                        created_ts = int(repo.last_modified) if hasattr(repo, "last_modified") else 0
                        models[repo.repo_id] = ModelInfo(
                            model_id=repo.repo_id,
                            path=model_path,
                            source="huggingface",
                            created=created_ts,
                            size_bytes=repo.size_on_disk,
                        )
                if models:
                    return models
        except Exception as e:
            logger.debug(f"huggingface_hub.scan_cache_dir could not be used: {e}. Falling back to directory scan.")

        # Filesystem fallback scan of cache_dir
        if not cache_dir.exists() or not cache_dir.is_dir():
            return models

        try:
            for entry in cache_dir.iterdir():
                if entry.is_dir() and entry.name.startswith("models--"):
                    # Format: models--<org>--<repo> or models--<repo>
                    parts = entry.name.split("--")[1:]
                    if not parts:
                        continue
                    repo_id = "/".join(parts)

                    # Look for snapshots
                    snapshots_dir = entry / "snapshots"
                    snapshot_path = entry
                    created_ts = int(entry.stat().st_mtime)

                    if snapshots_dir.is_dir():
                        snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
                        if snapshots:
                            # Sort by modification time to get latest
                            latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
                            snapshot_path = latest_snapshot
                            created_ts = int(latest_snapshot.stat().st_mtime)

                    # Verify if it contains model files (config.json or weight files)
                    if self._is_valid_model_dir(snapshot_path) or self._is_valid_model_dir(entry):
                        models[repo_id] = ModelInfo(
                            model_id=repo_id,
                            path=snapshot_path,
                            source="huggingface",
                            created=created_ts,
                        )
        except Exception as e:
            logger.error(f"Error scanning HF cache directory {cache_dir}: {e}")

        return models

    def scan_local_dirs(self) -> Dict[str, ModelInfo]:
        """Scan local / external directories for downloaded models."""
        models: Dict[str, ModelInfo] = {}

        for base_dir in settings.local_models_dirs:
            if not base_dir.exists() or not base_dir.is_dir():
                continue

            try:
                # First check if the directory itself is a model
                if self._is_valid_model_dir(base_dir):
                    models[base_dir.name] = ModelInfo(
                        model_id=base_dir.name,
                        path=base_dir,
                        source="local",
                        created=int(base_dir.stat().st_mtime),
                    )

                # Check 1st-level subdirectories: ./models/my-model
                for sub1 in base_dir.iterdir():
                    if not sub1.is_dir() or sub1.name.startswith("."):
                        continue

                    if self._is_valid_model_dir(sub1):
                        models[sub1.name] = ModelInfo(
                            model_id=sub1.name,
                            path=sub1,
                            source="local",
                            created=int(sub1.stat().st_mtime),
                        )
                        continue

                    # Check 2nd-level subdirectories: ./models/org/repo
                    for sub2 in sub1.iterdir():
                        if not sub2.is_dir() or sub2.name.startswith("."):
                            continue

                        if self._is_valid_model_dir(sub2):
                            rel_id = f"{sub1.name}/{sub2.name}"
                            models[rel_id] = ModelInfo(
                                model_id=rel_id,
                                path=sub2,
                                source="local",
                                created=int(sub2.stat().st_mtime),
                            )
            except Exception as e:
                logger.error(f"Error scanning local model directory {base_dir}: {e}")

        return models

    def _is_valid_model_dir(self, directory: Path) -> bool:
        """Check if directory contains Hugging Face model configuration or weights."""
        if not directory.exists() or not directory.is_dir():
            return False

        indicator_files = {
            "config.json",
            "tokenizer_config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "model_index.json",
        }

        try:
            for child in directory.iterdir():
                if child.name in indicator_files or child.name.endswith(".safetensors"):
                    return True
        except Exception:
            pass

        return False

    def list_models(self) -> List[ModelObject]:
        """
        Dynamically discovers all models from Hugging Face cache and local directories.
        Always rescans so externally downloaded models are immediately visible!
        """
        discovered: Dict[str, ModelInfo] = {}

        # Scan local directories first
        discovered.update(self.scan_local_dirs())

        # Scan HF cache
        discovered.update(self.scan_hf_cache())

        # Update cache
        self._cached_models = discovered

        # Convert to OpenAI ModelObject schemas
        model_objects = [info.to_model_object() for info in discovered.values()]
        # Sort by creation time descending
        model_objects.sort(key=lambda m: m.created, reverse=True)
        return model_objects

    def get_model(self, model_id: str) -> Optional[ModelObject]:
        """Get model object by ID or path."""
        # Rescan to ensure freshness
        models = self.list_models()
        for m in models:
            if m.id == model_id:
                return m
            # Also check if user passed short name (e.g. repo name without org)
            if "/" in m.id and m.id.split("/")[-1] == model_id:
                return m

        # Check if model_id is a direct local directory
        local_path = Path(model_id).resolve()
        if self._is_valid_model_dir(local_path):
            info = ModelInfo(
                model_id=model_id,
                path=local_path,
                source="local",
                created=int(local_path.stat().st_mtime),
            )
            return info.to_model_object()

        return None

    def resolve_model_path(self, model_id: str) -> str:
        """
        Resolves a requested model_id to either:
        1. A verified local directory path (if in local dirs or HF cache snapshot)
        2. The Hugging Face repo ID (for online/cached resolution)
        """
        # 1. Direct path check
        direct_path = Path(model_id).resolve()
        if self._is_valid_model_dir(direct_path):
            return str(direct_path)

        # 2. Rescan models
        self.list_models()

        # Exact match in discovered models
        if model_id in self._cached_models:
            info = self._cached_models[model_id]
            if info.path.exists():
                return str(info.path)
            return info.model_id

        # Short name match (e.g. "Qwen2.5-0.5B-Instruct" matching "Qwen/Qwen2.5-0.5B-Instruct")
        for key, info in self._cached_models.items():
            if "/" in key and key.split("/")[-1] == model_id:
                if info.path.exists():
                    return str(info.path)
                return info.model_id

        # If not found locally, return model_id as HF repo identifier
        return model_id


model_registry = ModelRegistry()
