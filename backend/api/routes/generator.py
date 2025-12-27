"""Generator API routes"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import asyncio
import json

from database.connection import get_db
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


class GenerationConstraints(BaseModel):
    min_mw: float = Field(200, ge=100, le=1000)
    max_mw: float = Field(600, ge=100, le=1000)
    min_chain_length: int = Field(8, ge=4, le=22)
    max_chain_length: int = Field(18, ge=4, le=22)
    require_quat: bool = True
    require_novel: bool = True
    allowed_counterions: list[str] = ["Cl", "Br", "I"]


class GenerationWeights(BaseModel):
    efficacy: float = Field(0.4, ge=0, le=1)
    safety: float = Field(0.3, ge=0, le=1)
    environmental: float = Field(0.2, ge=0, le=1)
    sa_score: float = Field(0.1, ge=0, le=1)


class GenerationRequest(BaseModel):
    num_molecules: int = Field(100, ge=1, le=10000)
    constraints: GenerationConstraints = GenerationConstraints()
    weights: GenerationWeights = GenerationWeights()
    batch_size: int = Field(64, ge=1, le=512)
    use_gpu: bool = True
    num_workers: int = Field(8, ge=1, le=64)


class GenerationStatus(BaseModel):
    is_running: bool
    molecules_generated: int
    molecules_per_hour: float
    pareto_frontier_size: int
    current_batch: int
    total_batches: int
    elapsed_seconds: float
    estimated_remaining_seconds: float
    top_scores: dict


@router.post("/start")
async def start_generation(request: GenerationRequest, background_tasks: BackgroundTasks, req: Request, db: AsyncSession = Depends(get_db)):
    generator = req.app.state.generator
    if generator.is_running:
        raise HTTPException(status_code=409, detail="Generation already in progress")
    total_weight = request.weights.efficacy + request.weights.safety + request.weights.environmental + request.weights.sa_score
    if abs(total_weight - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail=f"Weights must sum to 1.0 (got {total_weight})")
    background_tasks.add_task(generator.run_generation, num_molecules=request.num_molecules, constraints=request.constraints.model_dump(),
                              weights=request.weights.model_dump(), batch_size=request.batch_size, use_gpu=request.use_gpu, num_workers=request.num_workers)
    return {"status": "started", "target_molecules": request.num_molecules}


@router.post("/stop")
async def stop_generation(req: Request):
    generator = req.app.state.generator
    if not generator.is_running:
        raise HTTPException(status_code=400, detail="No generation in progress")
    await generator.stop()
    return {"status": "stopped", "molecules_generated": generator.molecules_generated}


@router.get("/status", response_model=GenerationStatus)
async def get_status(req: Request):
    generator = req.app.state.generator
    return GenerationStatus(is_running=generator.is_running, molecules_generated=generator.molecules_generated,
                           molecules_per_hour=generator.molecules_per_hour, pareto_frontier_size=generator.pareto_frontier_size,
                           current_batch=generator.current_batch, total_batches=generator.total_batches,
                           elapsed_seconds=generator.elapsed_seconds, estimated_remaining_seconds=generator.estimated_remaining_seconds,
                           top_scores=generator.top_scores)


@router.get("/stream")
async def stream_status(req: Request):
    generator = req.app.state.generator
    async def event_generator():
        while True:
            if not generator.is_running:
                yield f"data: {json.dumps({'event': 'stopped'})}\n\n"
                break
            status = {"event": "update", "molecules_generated": generator.molecules_generated,
                     "molecules_per_hour": generator.molecules_per_hour, "pareto_frontier_size": generator.pareto_frontier_size,
                     "top_scores": generator.top_scores}
            yield f"data: {json.dumps(status)}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


@router.post("/reset")
async def reset_generator(req: Request):
    generator = req.app.state.generator
    if generator.is_running:
        raise HTTPException(status_code=409, detail="Cannot reset while generation is running")
    await generator.reset()
    return {"status": "reset"}


@router.get("/pareto")
async def get_pareto_frontier(req: Request, db: AsyncSession = Depends(get_db)):
    generator = req.app.state.generator
    pareto_molecules = await generator.get_pareto_frontier()
    return {"count": len(pareto_molecules), "molecules": pareto_molecules}
