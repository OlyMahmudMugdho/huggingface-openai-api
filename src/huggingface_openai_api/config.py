import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class Settings(BaseModel):
    # Server configuration
    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or None)

    # Hugging Face cache directories
    hf_home: Path = Field(
        default_factory=lambda: Path(
            os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        ).resolve()
    )
    hf_hub_cache: Path = Field(
        default_factory=lambda: Path(
            os.getenv(
                "HF_HUB_CACHE",
                os.path.join(
                    os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                    "hub",
                ),
            )
        ).resolve()
    )

    # Custom / external model search paths
    # Users can set LOCAL_MODELS_DIR or MODELS_DIR (comma- or colon-separated)
    local_models_dirs: List[Path] = Field(
        default_factory=lambda: [
            Path(p.strip()).resolve()
            for raw in [os.getenv("LOCAL_MODELS_DIR"), os.getenv("MODELS_DIR"), "./models"]
            if raw
            for p in raw.replace(";", ":").replace(",", ":").split(":")
            if p.strip()
        ]
    )

    # Inference settings
    default_device: str = Field(
        default_factory=lambda: os.getenv("DEVICE", "auto")
    )
    torch_dtype: str = Field(
        default_factory=lambda: os.getenv("TORCH_DTYPE", "auto")
    )
    max_loaded_models: int = Field(
        default_factory=lambda: int(os.getenv("MAX_LOADED_MODELS", "1"))
    )
    default_max_new_tokens: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "512"))
    )
    trust_remote_code: bool = Field(
        default_factory=lambda: os.getenv("TRUST_REMOTE_CODE", "false").lower() in ("true", "1", "yes")
    )


settings = Settings()
