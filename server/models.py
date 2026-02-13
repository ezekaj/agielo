from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import uuid


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "cognitive"
    messages: List[ChatMessage]
    max_tokens: int = 4096
    temperature: float = 0.7
    stream: bool = False
    stop: Optional[List[str]] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "cognitive"
    choices: List[ChatCompletionChoice]
    usage: TokenUsage = Field(default_factory=TokenUsage)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "elo-cognitive"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class CognitiveStateResponse(BaseModel):
    confidence: float = 0.0
    system_used: str = "unknown"
    emotion: str = "neutral"
    emotion_intensity: float = 0.0
    memory_episodes: int = 0
    novelty_rate: float = 0.0
    messages_processed: int = 0
    uptime_seconds: float = 0.0


class EvolutionStateResponse(BaseModel):
    current_cycle: int = 0
    facts_this_cycle: int = 0
    facts_per_cycle: int = 100
    baseline_score: Optional[float] = None
    current_score: Optional[float] = None
    total_trainings: int = 0
    improvements: List[Dict[str, Any]] = []


class ObserveRequest(BaseModel):
    app_name: str
    window_title: str = ""
    url: str = ""
    bundle_id: str = ""
    ocr_text: str = ""
    timestamp: int = 0


class ConsolidateResponse(BaseModel):
    success: bool
    schemas_extracted: int = 0
    memories_consolidated: int = 0


class TopicResponse(BaseModel):
    name: str
    confidence: float
    curiosity: float
    learning_priority: float
    exposure_count: int
