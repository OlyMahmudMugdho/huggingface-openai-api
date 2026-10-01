import logging
from fastapi import APIRouter, HTTPException, status

from ..model_registry import model_registry
from ..schemas import ErrorResponse, ModelList, ModelObject

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Models"])


@router.get(
    "/models",
    response_model=ModelList,
    summary="List all available models",
    description="Dynamically scans Hugging Face cache and local external model directories to list all ready-to-serve models.",
)
async def list_models() -> ModelList:
    models = model_registry.list_models()
    return ModelList(object="list", data=models)


@router.get(
    "/models/{model_id:path}",
    response_model=ModelObject,
    summary="Retrieve a model instance",
    description="Retrieves a model instance by ID or path.",
    responses={404: {"model": ErrorResponse}},
)
async def get_model(model_id: str) -> ModelObject:
    model = model_registry.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model '{model_id}' does not exist or has not been downloaded.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )
    return model
