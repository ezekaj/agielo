"""
Human Cognition AI - FastAPI Backend

Serves the cognitive simulation engine as a REST API.
Endpoints for simulation, chat, module inspection, and text analysis.
"""

import sys
import os
import asyncio
import time
import logging
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.simulator import simulator, PHASES, MODULE_DEFINITIONS, SCENARIOS, _serialize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

ALLOWED_ORIGINS = [
    "https://ezekaj.github.io",
    "http://localhost:3000",
]

SIMULATION_TIMEOUT_SECONDS = 30

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Human Cognition AI",
    description="A neuroscience-based cognitive architecture API. 15+ cognitive modules implementing brain-realistic cognition across 5 phases.",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

startup_time = time.time()

VALID_SCENARIOS = list(SCENARIOS.keys())


class SimulateRequest(BaseModel):
    scenario: str = Field(
        ...,
        description="Scenario key to simulate",
        examples=["sentence_processing", "memory_recall", "financial_decision", "word_learning", "danger_response"],
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional parameters to override defaults (e.g. dim, speed multipliers)",
    )

    @field_validator("scenario")
    @classmethod
    def validate_scenario(cls, v: str) -> str:
        if v not in VALID_SCENARIOS:
            raise ValueError(f"Unknown scenario '{v}'. Must be one of: {VALID_SCENARIOS}")
        return v

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is None:
            return v
        if "dim" in v:
            dim = v["dim"]
            if not isinstance(dim, int) or dim < 1 or dim > 512:
                raise ValueError("dim must be an integer between 1 and 512")
        return v


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=5000, description="User message to process through the cognitive framework")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Conversation history")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., max_length=5000, description="Text to analyze through cognitive lenses")
    lenses: Optional[List[str]] = Field(
        default=None,
        description="Cognitive lenses to apply",
        examples=[["memory", "emotion", "reasoning", "creativity"]],
    )


@app.get("/api/health")
@limiter.limit("30/minute")
async def health(request: Request):
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - startup_time, 1),
        "agent_available": simulator._agent is not None,
    }


@app.get("/api/info")
@limiter.limit("30/minute")
async def info(request: Request):
    simulator._ensure_agent()
    agent_modules = list(simulator._agent._modules.keys()) if simulator._agent else []
    return {
        "name": "Human Cognition AI (AGiELO)",
        "version": "1.0.0",
        "description": "A neuroscience-based cognitive architecture implementing 15+ brain-realistic modules across 5 phases.",
        "phase_count": len(PHASES),
        "module_count": len(MODULE_DEFINITIONS),
        "scenario_count": len(SCENARIOS),
        "available_scenarios": list(SCENARIOS.keys()),
        "active_modules": agent_modules,
        "capabilities": [
            "Predictive Coding (Free Energy Principle)",
            "Dual-Process Thinking (System 1/2)",
            "Emotional Processing with Somatic Markers",
            "Multi-Store Memory (Episodic, Semantic, Procedural, Working)",
            "Curiosity-Driven Learning",
            "Theory of Mind",
            "Sleep-Based Memory Consolidation",
            "Creative Conceptual Blending",
            "Metacognitive Self-Awareness",
        ],
        "theories": [
            "Free Energy Principle (Friston 2010)",
            "Dual-Process Theory (Kahneman 2011)",
            "Somatic Marker Hypothesis (Damasio 1994)",
            "Global Workspace Theory (Baars 1988)",
            "Predictive Coding (Rao & Ballard 1999)",
        ],
    }


@app.get("/api/modules")
@limiter.limit("30/minute")
async def get_modules(request: Request):
    simulator._ensure_agent()
    modules = simulator.get_modules()
    return JSONResponse(content=_serialize({"modules": modules, "count": len(modules)}))


@app.get("/api/phases")
@limiter.limit("30/minute")
async def get_phases(request: Request):
    phases = simulator.get_phases()
    return JSONResponse(content=_serialize({"phases": phases, "count": len(phases)}))


@app.post("/api/simulate")
@limiter.limit("5/minute")
async def simulate(request: Request, body: SimulateRequest):
    logger.info("Simulation requested: scenario=%s", body.scenario)
    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, simulator.simulate, body.scenario, body.parameters
            ),
            timeout=SIMULATION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error("Simulation timed out after %ds: scenario=%s", SIMULATION_TIMEOUT_SECONDS, body.scenario)
        raise HTTPException(status_code=504, detail=f"Simulation timed out after {SIMULATION_TIMEOUT_SECONDS}s")
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    logger.info("Simulation completed: scenario=%s, engine=%s", body.scenario, result.get("engine"))
    return JSONResponse(content=_serialize(result))


@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    logger.info("Chat requested: message_length=%d", len(body.message))
    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, simulator.chat, body.message, body.history
            ),
            timeout=SIMULATION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error("Chat timed out after %ds", SIMULATION_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail=f"Chat processing timed out after {SIMULATION_TIMEOUT_SECONDS}s")
    logger.info("Chat completed: engine=%s", result.get("engine"))
    return JSONResponse(content=_serialize(result))


@app.post("/api/analyze")
@limiter.limit("10/minute")
async def analyze(request: Request, body: AnalyzeRequest):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    logger.info("Analysis requested: text_length=%d, lenses=%s", len(body.text), body.lenses)
    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, simulator.analyze_text, body.text, body.lenses
            ),
            timeout=SIMULATION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error("Analysis timed out after %ds", SIMULATION_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail=f"Analysis timed out after {SIMULATION_TIMEOUT_SECONDS}s")
    logger.info("Analysis completed: engine=%s", result.get("engine"))
    return JSONResponse(content=_serialize(result))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
