import uvicorn
from .app import app
from .config import settings


def main() -> None:
    """Run the Hugging Face OpenAI-compatible API server."""
    uvicorn.run(
        "huggingface_openai_api.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


__all__ = ["app", "main"]
