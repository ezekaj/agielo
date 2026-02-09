"""
Human Cognition AI - FastAPI Backend

Serves the cognitive simulation engine as a REST API.
Endpoints for simulation, chat, module inspection, and text analysis.
"""

import sys
import os
import time
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.simulator import simulator, PHASES, MODULE_DEFINITIONS, SCENARIOS

app = FastAPI(
    title="Human Cognition AI",
    description="A neuroscience-based cognitive architecture API. 15+ cognitive modules implementing brain-realistic cognition across 5 phases.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

startup_time = time.time()


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


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to process through the cognitive framework")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Conversation history")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Text to analyze through cognitive lenses")
    lenses: Optional[List[str]] = Field(
        default=None,
        description="Cognitive lenses to apply",
        examples=[["memory", "emotion", "reasoning", "creativity"]],
    )


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - startup_time, 1),
        "agent_available": simulator._agent is not None,
    }


@app.get("/api/info")
async def info():
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
async def get_modules():
    simulator._ensure_agent()
    modules = simulator.get_modules()
    return {"modules": modules, "count": len(modules)}


@app.get("/api/phases")
async def get_phases():
    phases = simulator.get_phases()
    return {"phases": phases, "count": len(phases)}


@app.post("/api/simulate")
async def simulate(request: SimulateRequest):
    result = simulator.simulate(request.scenario, request.parameters)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    result = simulator.chat(request.message, request.history)
    return result


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    result = simulator.analyze_text(request.text, request.lenses)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
