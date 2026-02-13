from fastapi import APIRouter
from typing import List

from server.models import (
    CognitiveStateResponse,
    EvolutionStateResponse,
    ObserveRequest,
    ConsolidateResponse,
    TopicResponse,
)

router = APIRouter(prefix="/api/v1/cognitive")


def get_engine():
    from server.cognitive_server import engine
    return engine


@router.get("/state", response_model=CognitiveStateResponse)
async def cognitive_state():
    eng = get_engine()
    if not eng.ai:
        import time
        return CognitiveStateResponse(
            uptime_seconds=time.time() - eng.start_time,
        )
    stats = eng.ai.get_stats()
    introspection = eng.ai.introspect()

    emotion = "neutral"
    emotion_intensity = 0.0
    if hasattr(eng, 'emotions') and eng.emotions:
        try:
            state = eng.emotions.get_current_state()
            if state:
                emotion = state.get('dominant_emotion', 'neutral')
                emotion_intensity = state.get('intensity', 0.0)
        except Exception:
            pass

    last_context = {}
    if eng.ai.history:
        last_context = eng.ai.history[-1].get('context', {})

    import time
    return CognitiveStateResponse(
        confidence=last_context.get('confidence', 0.5),
        system_used=last_context.get('system_used', 'system1'),
        emotion=emotion,
        emotion_intensity=emotion_intensity,
        memory_episodes=stats.get('memory', {}).get('total_events', 0),
        novelty_rate=stats.get('memory', {}).get('novelty_rate', 0.0),
        messages_processed=stats.get('messages_processed', 0),
        uptime_seconds=time.time() - eng.start_time,
    )


@router.get("/introspect")
async def cognitive_introspect():
    eng = get_engine()
    if not eng.ai:
        return {"status": "initializing"}
    return eng.ai.introspect()


@router.get("/stats")
async def cognitive_stats():
    eng = get_engine()
    if not eng.ai:
        return {"status": "initializing"}
    stats = eng.ai.get_stats()

    if eng.trainer:
        try:
            trainer_stats = eng.trainer.get_stats()
            stats['knowledge'] = trainer_stats
        except Exception:
            pass

    if eng.evolution:
        stats['evolution'] = eng.evolution.state

    return stats


@router.get("/evolution", response_model=EvolutionStateResponse)
async def cognitive_evolution():
    eng = get_engine()
    if not eng.evolution:
        return EvolutionStateResponse()

    state = eng.evolution.state
    return EvolutionStateResponse(
        current_cycle=state.get('current_cycle', 0),
        facts_this_cycle=state.get('facts_this_cycle', 0),
        facts_per_cycle=state.get('facts_per_cycle', 100),
        baseline_score=state.get('baseline_score'),
        current_score=state.get('current_score'),
        total_trainings=state.get('total_trainings', 0),
        improvements=state.get('improvements', []),
    )


@router.get("/topics", response_model=List[TopicResponse])
async def cognitive_topics():
    eng = get_engine()
    if not eng.active_learner:
        return []

    try:
        recs = eng.active_learner.get_learning_recommendations(k=10)
        topics = []
        for name, priority, _reason in recs:
            topic_obj = eng.active_learner.topics.get(name)
            if topic_obj:
                topics.append(TopicResponse(
                    name=name,
                    confidence=topic_obj.confidence,
                    curiosity=topic_obj.curiosity,
                    learning_priority=priority,
                    exposure_count=topic_obj.exposure_count,
                ))
        return topics
    except Exception:
        return []


@router.post("/consolidate", response_model=ConsolidateResponse)
async def cognitive_consolidate():
    eng = get_engine()
    result = eng.ai.consolidate()
    return ConsolidateResponse(
        success=result.get('enabled', True),
        schemas_extracted=result.get('schemas_extracted', 0),
        memories_consolidated=result.get('memories_consolidated', 0),
    )


@router.post("/observe")
async def cognitive_observe(request: ObserveRequest):
    eng = get_engine()
    text = f"{request.app_name}: {request.window_title}"
    if request.url:
        text += f" ({request.url})"

    try:
        emb = eng.ai._text_to_embedding(text)
        eng.ai.cognition.perceive(emb)

        if eng.active_learner:
            topic = request.app_name.lower()
            eng.active_learner.boost_curiosity(topic, 0.05)

        return {"status": "ok", "perceived": text}
    except Exception as e:
        return {"status": "error", "error": str(e)}
