import asyncio
import logging
from typing import Union
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from ..engine import inference_engine
from ..schemas import CompletionRequest, CompletionResponse, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Completions"])


@router.post(
    "/completions",
    response_model=Union[CompletionResponse, None],
    summary="Create a text completion",
    description="Creates a completion for the provided prompt and parameters in OpenAI format.",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_completion(
    request: CompletionRequest,
):
    try:
        if request.stream:
            return StreamingResponse(
                inference_engine.generate_completion_stream(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                },
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            inference_engine.generate_completion,
            request,
        )
        return response

    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"Model '{request.model}' could not be found locally or downloaded: {str(e)}",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )
    except Exception as e:
        logger.exception(f"Error during completion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Generation failed: {str(e)}",
                    "type": "server_error",
                    "param": None,
                    "code": "internal_error",
                }
            },
        )
