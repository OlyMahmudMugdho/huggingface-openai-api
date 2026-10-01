import os
import sys
from pathlib import Path
import uvicorn

# Automatically add src directory to python path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from huggingface_openai_api.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "huggingface_openai_api.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        app_dir=str(SRC_DIR),
    )
