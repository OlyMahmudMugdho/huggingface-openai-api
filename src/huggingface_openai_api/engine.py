import asyncio
import gc
import logging
import threading
import time
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

from .config import settings
from .model_registry import model_registry
from .schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionChunk,
    CompletionChunkChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    DeltaMessage,
    Usage,
)

logger = logging.getLogger(__name__)


def get_optimal_device() -> str:
    """Determine the optimal compute device."""
    if settings.default_device != "auto":
        return settings.default_device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_torch_dtype():
    """Determine the torch dtype to use."""
    import torch

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if settings.torch_dtype in dtype_map:
        return dtype_map[settings.torch_dtype]

    # auto
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


class ModelInstance:
    """Holds a loaded model and tokenizer in memory."""

    def __init__(self, model_id: str, resolved_path: str, model, tokenizer, device: str):
        self.model_id = model_id
        self.resolved_path = resolved_path
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.last_used = time.time()
        self.lock = threading.Lock()

    def touch(self):
        self.last_used = time.time()


class InferenceEngine:
    """Manages loaded Hugging Face models and generation."""

    def __init__(self):
        self._loaded_models: Dict[str, ModelInstance] = {}
        self._management_lock = threading.Lock()

    def get_or_load_model(self, model_id: str) -> ModelInstance:
        """Loads model into memory if not loaded yet, or returns cached instance."""
        with self._management_lock:
            # Check if already loaded
            if model_id in self._loaded_models:
                instance = self._loaded_models[model_id]
                instance.touch()
                return instance

            # Check if resolved path is loaded under another key
            resolved_path = model_registry.resolve_model_path(model_id)
            for inst in self._loaded_models.values():
                if inst.resolved_path == resolved_path:
                    inst.touch()
                    self._loaded_models[model_id] = inst
                    return inst

            # Evict if exceeding max_loaded_models
            if len(self._loaded_models) >= settings.max_loaded_models:
                self._evict_oldest_model()

            logger.info(f"Loading model '{model_id}' from path '{resolved_path}'...")

            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            device = get_optimal_device()
            dtype = get_torch_dtype()

            tokenizer = AutoTokenizer.from_pretrained(
                resolved_path,
                trust_remote_code=settings.trust_remote_code,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

            model_kwargs = {
                "trust_remote_code": settings.trust_remote_code,
                "torch_dtype": dtype,
            }

            if device == "cuda":
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = None

            model = AutoModelForCausalLM.from_pretrained(
                resolved_path,
                **model_kwargs,
            )

            if device != "cuda":
                model.to(device)

            model.eval()

            instance = ModelInstance(
                model_id=model_id,
                resolved_path=resolved_path,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            self._loaded_models[model_id] = instance
            logger.info(f"Successfully loaded model '{model_id}' on {device}.")
            return instance

    def _evict_oldest_model(self):
        """Evicts the least recently used model from memory."""
        if not self._loaded_models:
            return

        oldest_key = min(self._loaded_models.keys(), key=lambda k: self._loaded_models[k].last_used)
        logger.info(f"Evicting model '{oldest_key}' from memory.")
        instance = self._loaded_models.pop(oldest_key)
        del instance.model
        del instance.tokenizer
        del instance

        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def format_chat_prompt(self, instance: ModelInstance, messages: List[ChatMessage]) -> str:
        """Formats chat messages using tokenizer chat template or standard fallback."""
        tokenizer = instance.tokenizer

        dict_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Try tokenizer's apply_chat_template
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                return tokenizer.apply_chat_template(
                    dict_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.warning(f"Error applying chat template: {e}. Falling back to default format.")

        # Fallback formatting
        lines = []
        for msg in messages:
            role_title = msg.role.capitalize()
            lines.append(f"{role_title}: {msg.content}")
        lines.append("Assistant:")
        return "\n".join(lines) + " "

    def generate_chat(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Non-streaming chat completion."""
        instance = self.get_or_load_model(request.model)
        prompt = self.format_chat_prompt(instance, request.messages)

        with instance.lock:
            inputs = instance.tokenizer(prompt, return_tensors="pt")
            if instance.device != "cuda" or instance.model.device.type != "cuda":
                inputs = inputs.to(instance.device)
            else:
                inputs = inputs.to(instance.model.device)

            prompt_token_count = inputs.input_ids.shape[1]
            max_new_tokens = request.get_max_tokens(settings.default_max_new_tokens)

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": instance.tokenizer.pad_token_id,
                "eos_token_id": instance.tokenizer.eos_token_id,
            }

            if request.temperature and request.temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = request.temperature
                if request.top_p:
                    gen_kwargs["top_p"] = request.top_p
            else:
                gen_kwargs["do_sample"] = False

            if request.repetition_penalty:
                gen_kwargs["repetition_penalty"] = request.repetition_penalty

            outputs = instance.model.generate(**inputs, **gen_kwargs)
            gen_tokens = outputs[0][prompt_token_count:]
            completion_token_count = len(gen_tokens)
            output_text = instance.tokenizer.decode(gen_tokens, skip_special_tokens=True)

            # Handle stop sequences if any
            finish_reason = "stop"
            if completion_token_count >= max_new_tokens:
                finish_reason = "length"

            if request.stop:
                stops = [request.stop] if isinstance(request.stop, str) else request.stop
                for stop_str in stops:
                    if stop_str and stop_str in output_text:
                        output_text = output_text.split(stop_str)[0]
                        finish_reason = "stop"

            req_id = f"chatcmpl-{int(time.time()*1000)}"
            return ChatCompletionResponse(
                id=req_id,
                model=request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=output_text),
                        finish_reason=finish_reason,
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_token_count,
                    completion_tokens=completion_token_count,
                    total_tokens=prompt_token_count + completion_token_count,
                ),
            )

    async def generate_chat_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat completion yielding Server-Sent Events (SSE)."""
        instance = self.get_or_load_model(request.model)
        prompt = self.format_chat_prompt(instance, request.messages)

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            instance.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        inputs = instance.tokenizer(prompt, return_tensors="pt")
        if instance.device != "cuda" or instance.model.device.type != "cuda":
            inputs = inputs.to(instance.device)
        else:
            inputs = inputs.to(instance.model.device)

        max_new_tokens = request.get_max_tokens(settings.default_max_new_tokens)
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            pad_token_id=instance.tokenizer.pad_token_id,
            eos_token_id=instance.tokenizer.eos_token_id,
        )

        if request.temperature and request.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = request.temperature
            if request.top_p:
                gen_kwargs["top_p"] = request.top_p
        else:
            gen_kwargs["do_sample"] = False

        if request.repetition_penalty:
            gen_kwargs["repetition_penalty"] = request.repetition_penalty

        req_id = f"chatcmpl-{int(time.time()*1000)}"

        # Initial chunk specifying role
        first_chunk = ChatCompletionChunk(
            id=req_id,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=""),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        # Run generation in a separate thread
        thread = threading.Thread(target=instance.model.generate, kwargs=gen_kwargs)
        thread.start()

        loop = asyncio.get_event_loop()
        stop_seqs = [request.stop] if isinstance(request.stop, str) else (request.stop or [])

        def get_next_token():
            try:
                return next(streamer)
            except StopIteration:
                return None

        accumulated_text = ""
        stopped = False

        while True:
            text = await loop.run_in_executor(None, get_next_token)
            if text is None:
                break

            accumulated_text += text
            if stop_seqs:
                for stop_seq in stop_seqs:
                    if stop_seq and stop_seq in accumulated_text:
                        stopped = True
                        break
            if stopped:
                break

            chunk = ChatCompletionChunk(
                id=req_id,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=DeltaMessage(content=text),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Final chunk
        final_chunk = ChatCompletionChunk(
            id=req_id,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    def generate_completion(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Non-streaming text completion."""
        instance = self.get_or_load_model(request.model)
        prompt = request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt)

        with instance.lock:
            inputs = instance.tokenizer(prompt, return_tensors="pt")
            if instance.device != "cuda" or instance.model.device.type != "cuda":
                inputs = inputs.to(instance.device)
            else:
                inputs = inputs.to(instance.model.device)

            prompt_token_count = inputs.input_ids.shape[1]
            max_new_tokens = request.max_tokens or 128

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": instance.tokenizer.pad_token_id,
                "eos_token_id": instance.tokenizer.eos_token_id,
            }

            if request.temperature and request.temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = request.temperature
                if request.top_p:
                    gen_kwargs["top_p"] = request.top_p
            else:
                gen_kwargs["do_sample"] = False

            if request.repetition_penalty:
                gen_kwargs["repetition_penalty"] = request.repetition_penalty

            outputs = instance.model.generate(**inputs, **gen_kwargs)
            gen_tokens = outputs[0][prompt_token_count:]
            completion_token_count = len(gen_tokens)
            output_text = instance.tokenizer.decode(gen_tokens, skip_special_tokens=True)

            finish_reason = "stop"
            if completion_token_count >= max_new_tokens:
                finish_reason = "length"

            if request.stop:
                stops = [request.stop] if isinstance(request.stop, str) else request.stop
                for stop_str in stops:
                    if stop_str and stop_str in output_text:
                        output_text = output_text.split(stop_str)[0]
                        finish_reason = "stop"

            req_id = f"cmpl-{int(time.time()*1000)}"
            return CompletionResponse(
                id=req_id,
                model=request.model,
                choices=[
                    CompletionChoice(
                        text=output_text,
                        index=0,
                        finish_reason=finish_reason,
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_token_count,
                    completion_tokens=completion_token_count,
                    total_tokens=prompt_token_count + completion_token_count,
                ),
            )

    async def generate_completion_stream(
        self,
        request: CompletionRequest,
    ) -> AsyncGenerator[str, None]:
        """Streaming text completion yielding Server-Sent Events (SSE)."""
        instance = self.get_or_load_model(request.model)
        prompt = request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt)

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            instance.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        inputs = instance.tokenizer(prompt, return_tensors="pt")
        if instance.device != "cuda" or instance.model.device.type != "cuda":
            inputs = inputs.to(instance.device)
        else:
            inputs = inputs.to(instance.model.device)

        max_new_tokens = request.max_tokens or 128
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            pad_token_id=instance.tokenizer.pad_token_id,
            eos_token_id=instance.tokenizer.eos_token_id,
        )

        if request.temperature and request.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = request.temperature
            if request.top_p:
                gen_kwargs["top_p"] = request.top_p
        else:
            gen_kwargs["do_sample"] = False

        if request.repetition_penalty:
            gen_kwargs["repetition_penalty"] = request.repetition_penalty

        req_id = f"cmpl-{int(time.time()*1000)}"

        thread = threading.Thread(target=instance.model.generate, kwargs=gen_kwargs)
        thread.start()

        loop = asyncio.get_event_loop()
        stop_seqs = [request.stop] if isinstance(request.stop, str) else (request.stop or [])

        def get_next_token():
            try:
                return next(streamer)
            except StopIteration:
                return None

        accumulated_text = ""
        stopped = False

        while True:
            text = await loop.run_in_executor(None, get_next_token)
            if text is None:
                break

            accumulated_text += text
            if stop_seqs:
                for stop_seq in stop_seqs:
                    if stop_seq and stop_seq in accumulated_text:
                        stopped = True
                        break
            if stopped:
                break

            chunk = CompletionChunk(
                id=req_id,
                model=request.model,
                choices=[
                    CompletionChunkChoice(
                        text=text,
                        index=0,
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        final_chunk = CompletionChunk(
            id=req_id,
            model=request.model,
            choices=[
                CompletionChunkChoice(
                    text="",
                    index=0,
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


inference_engine = InferenceEngine()
