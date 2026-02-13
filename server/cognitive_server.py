import os
import sys
import time
import json
import logging
import threading
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    TokenUsage,
    ModelInfo,
    ModelListResponse,
)
from server.cognitive_routes import router as cognitive_router

logger = logging.getLogger("elo-cognitive")

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "qwen/qwen3-vl-8b")
COGNITIVE_PORT = int(os.environ.get("COGNITIVE_PORT", "1338"))


class CognitiveEngine:
    """Wraps agielo's cognitive system as a singleton service."""

    def __init__(self):
        self.start_time = time.time()
        self.ai = None
        self.evolution = None
        self.active_learner = None
        self.trainer = None
        self.emotions = None
        self.ready = False
        self._init_error = None

    def initialize(self):
        """Initialize cognitive modules in background thread so server starts fast."""
        def _init():
            try:
                # Pre-create a mock neuro_memory_adapter to prevent filesystem
                # timeouts when importing neuro_memory's heavy modules on macOS
                import types
                mock_adapter = types.ModuleType('integrations.neuro_memory_adapter')
                mock_adapter.NEURO_MEMORY_AVAILABLE = False
                mock_adapter.NeuroMemorySystem = None
                sys.modules['integrations.neuro_memory_adapter'] = mock_adapter

                from integrations.cognitive_ollama import CognitiveOllama

                self.ai = CognitiveOllama(
                    model="cognitive-local",
                    ollama_url=f"{LM_STUDIO_URL}/v1/chat/completions",
                )
                logger.info("CognitiveOllama initialized")
            except Exception as e:
                logger.error(f"CognitiveOllama init failed: {e}")
                self._init_error = str(e)

            try:
                from integrations.self_evolution import SelfEvolution
                self.evolution = SelfEvolution()
                logger.info("SelfEvolution loaded")
            except Exception as e:
                logger.warning(f"SelfEvolution not available: {e}")

            try:
                from integrations.active_learning import ActiveLearner
                self.active_learner = ActiveLearner()
                logger.info("ActiveLearner loaded")
            except Exception as e:
                logger.warning(f"ActiveLearner not available: {e}")

            try:
                from integrations.self_training import SelfTrainer
                self.trainer = SelfTrainer()
                logger.info("SelfTrainer loaded")
            except Exception as e:
                logger.warning(f"SelfTrainer not available: {e}")

            try:
                from phase3_motivation.emotion_system import EmotionSystem
                self.emotions = EmotionSystem()
                logger.info("EmotionSystem loaded")
            except Exception as e:
                logger.warning(f"EmotionSystem not available: {e}")

            self.ready = True
            logger.info("Cognitive engine fully initialized")

        thread = threading.Thread(target=_init, daemon=True)
        thread.start()
        logger.info("Cognitive engine initializing in background...")


engine = CognitiveEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.initialize()
    logger.info(f"Server ready (cognitive loading in background, LM Studio: {LM_STUDIO_URL})")
    yield
    logger.info("Cognitive engine shutting down")


app = FastAPI(title="ELO Cognitive Server", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cognitive_router)


@app.get("/health")
async def health():
    return {
        "status": "ok" if engine.ready else "initializing",
        "uptime": f"{time.time() - engine.start_time:.0f}s",
        "version": "0.1.0",
        "lm_studio": LM_STUDIO_URL,
        "cognitive": engine.ai is not None,
        "ready": engine.ready,
        "init_error": engine._init_error,
    }


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    models = [
        ModelInfo(id="cognitive", owned_by="elo-cognitive"),
        ModelInfo(id="cognitive-enhanced", owned_by="elo-cognitive"),
    ]
    return ModelListResponse(data=models)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    user_msg = ""
    system_msg = ""
    messages_for_forward = []

    for msg in request.messages:
        if msg.role == "system":
            system_msg = msg.content
        elif msg.role == "user":
            user_msg = msg.content
        messages_for_forward.append({"role": msg.role, "content": msg.content})

    cognitive_context = ""
    if engine.ai and user_msg:
        try:
            ctx = engine.ai._get_cognitive_context(user_msg)

            cognitive_context = "\n[Cognitive State]\n"
            cognitive_context += f"- Confidence: {ctx.get('confidence', 0.5):.0%}\n"
            cognitive_context += f"- Processing: {ctx.get('system_used', 'system1')}\n"

            mem = ctx.get('memory', {})
            if mem.get('is_novel'):
                cognitive_context += "- This is a novel/important topic\n"
            if mem.get('recalled'):
                cognitive_context += f"- Related memories: {len(mem['recalled'])} recalled\n"
                for i, m in enumerate(mem['recalled'][:2]):
                    cognitive_context += f"  - Memory {i+1}: {m[:100]}\n"

            if engine.active_learner:
                try:
                    words = user_msg.lower().split()
                    for word in words[:5]:
                        if len(word) > 3:
                            engine.active_learner.boost_curiosity(word, 0.02)
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"Cognitive processing error: {e}")

    if cognitive_context and messages_for_forward:
        if messages_for_forward[0]["role"] == "system":
            messages_for_forward[0]["content"] += cognitive_context
        else:
            messages_for_forward.insert(0, {
                "role": "system",
                "content": cognitive_context,
            })

    if request.stream:
        return StreamingResponse(
            _stream_from_lm_studio(request, messages_for_forward),
            media_type="text/event-stream",
        )

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            forward_payload = {
                "model": LM_STUDIO_MODEL,
                "messages": messages_for_forward,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": False,
            }
            if request.stop:
                forward_payload["stop"] = request.stop

            resp = await client.post(
                f"{LM_STUDIO_URL}/v1/chat/completions",
                json=forward_payload,
            )
            resp.raise_for_status()
            lm_response = resp.json()

            content = lm_response["choices"][0]["message"]["content"]

            if engine.ai and user_msg:
                try:
                    response_emb = engine.ai._text_to_embedding(content)
                    engine.ai.cognition.learn({
                        'input': engine.ai._text_to_embedding(user_msg),
                        'response': response_emb,
                        'reward': 0.5,
                    })
                    engine.ai.history.append({
                        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'user': user_msg,
                        'assistant': content,
                        'context': {
                            'confidence': cognitive_context.count('Confidence'),
                            'system_used': 'system1',
                        },
                    })
                    engine.ai.messages_processed += 1
                except Exception as e:
                    logger.warning(f"Post-response learning error: {e}")

            return ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessage(role="assistant", content=content),
                    )
                ],
                usage=TokenUsage(
                    prompt_tokens=lm_response.get("usage", {}).get("prompt_tokens", 0),
                    completion_tokens=lm_response.get("usage", {}).get("completion_tokens", 0),
                    total_tokens=lm_response.get("usage", {}).get("total_tokens", 0),
                ),
            )

        except httpx.ConnectError:
            return ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessage(
                            role="assistant",
                            content=f"[LM Studio not reachable at {LM_STUDIO_URL}. Please start LM Studio.]",
                        ),
                    )
                ],
            )
        except Exception as e:
            return ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessage(
                            role="assistant",
                            content=f"[Error forwarding to LM Studio: {e}]",
                        ),
                    )
                ],
            )


async def _stream_from_lm_studio(request: ChatCompletionRequest, messages: list):
    """Stream SSE chunks from LM Studio back to the client."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            forward_payload = {
                "model": LM_STUDIO_MODEL,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": True,
            }
            if request.stop:
                forward_payload["stop"] = request.stop

            async with client.stream(
                "POST",
                f"{LM_STUDIO_URL}/v1/chat/completions",
                json=forward_payload,
            ) as resp:
                full_content = ""
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        chunk_data = line[6:]
                        if chunk_data.strip() == "[DONE]":
                            yield f"data: [DONE]\n\n"
                            break
                        try:
                            chunk = json.loads(chunk_data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta:
                                full_content += delta["content"]
                            yield f"data: {chunk_data}\n\n"
                        except json.JSONDecodeError:
                            yield f"data: {chunk_data}\n\n"

                if engine.ai and full_content:
                    try:
                        user_msg = ""
                        for m in messages:
                            if m["role"] == "user":
                                user_msg = m["content"]
                        if user_msg:
                            response_emb = engine.ai._text_to_embedding(full_content)
                            engine.ai.cognition.learn({
                                'input': engine.ai._text_to_embedding(user_msg),
                                'response': response_emb,
                                'reward': 0.5,
                            })
                            engine.ai.messages_processed += 1
                    except Exception:
                        pass

        except httpx.ConnectError:
            error_chunk = {
                "choices": [{"delta": {"content": f"[LM Studio not reachable at {LM_STUDIO_URL}]"}, "index": 0}]
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_chunk = {
                "choices": [{"delta": {"content": f"[Stream error: {e}]"}, "index": 0}]
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=COGNITIVE_PORT)
