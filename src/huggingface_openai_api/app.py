import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .model_registry import model_registry
from .routes import chat, completions, models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("huggingface_openai_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Discover models and print summary
    logger.info("Initializing Hugging Face OpenAI-compatible API server...")
    logger.info(f"Hugging Face hub cache dir: {settings.hf_hub_cache}")
    logger.info(f"Local models search directories: {settings.local_models_dirs}")

    initial_models = model_registry.list_models()
    logger.info(f"Discovered {len(initial_models)} model(s) available for serving:")
    for m in initial_models:
        logger.info(f" - {m.id} ({m.owned_by}) at {m.location}")

    yield

    logger.info("Shutting down Hugging Face OpenAI-compatible API server...")


app = FastAPI(
    title="Hugging Face OpenAI-Compatible API",
    description="A FastAPI application that serves Hugging Face models (both cached and externally downloaded) using the OpenAI API standard.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for compatibility with web clients (OpenWebUI, LibreChat, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Optional API Key Authentication Middleware
@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if settings.api_key:
        # Exclude docs and health endpoints from API key check
        if request.url.path not in ["/", "/health", "/docs", "/openapi.json", "/redoc"]:
            auth_header = request.headers.get("Authorization")
            expected_bearer = f"Bearer {settings.api_key}"
            if not auth_header or auth_header != expected_bearer:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "error": {
                            "message": "Incorrect API key provided.",
                            "type": "invalid_request_error",
                            "param": None,
                            "code": "invalid_api_key",
                        }
                    },
                )
    return await call_next(request)


# Standard OpenAI error formatting for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_msg = "; ".join([f"{err['loc']}: {err['msg']}" for err in exc.errors()])
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": error_msg,
                "type": "invalid_request_error",
                "param": None,
                "code": "validation_error",
            }
        },
    )


# Standard OpenAI error formatting for HTTPExceptions
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )


# Include Routers
app.include_router(models.router)
app.include_router(chat.router)
app.include_router(completions.router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "name": "Hugging Face OpenAI-Compatible API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": [
            "/v1/models",
            "/v1/chat/completions",
            "/v1/completions",
            "/docs",
        ],
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
