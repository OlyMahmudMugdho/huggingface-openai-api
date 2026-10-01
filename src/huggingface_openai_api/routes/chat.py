import asyncio
import logging
from typing import Union
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from ..engine import inference_engine
from ..schemas import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Chat"])


@router.post(
    "/chat/completions",
    response_model=Union[ChatCompletionResponse, None],
    summary="Create a chat completion",
    description="Creates a model response for the given chat conversation in OpenAI format.",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_chat_completion(
    request: ChatCompletionRequest,
):
    try:
        if request.stream:
            return StreamingResponse(
                inference_engine.generate_chat_stream(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                },
            )

        # Run blocking inference in threadpool so event loop remains unblocked
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            inference_engine.generate_chat,
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
        logger.exception(f"Error during chat completion: {e}")
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
