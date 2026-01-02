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


@router.post("/fetch-chembl")
async def fetch_chembl_data(background_tasks: BackgroundTasks, req: Request, force_refresh: bool = False):
    """
    Fetch quaternary ammonium compound data from ChEMBL database.

    This fetches real experimental MIC data for known quaternary ammonium
    disinfectants and searches for additional compounds with antimicrobial activity.
    """
    chembl_fetcher = getattr(req.app.state, "chembl_fetcher", None)

    if not chembl_fetcher:
        raise HTTPException(status_code=503, detail="ChEMBL fetcher not initialized")

    # Run fetch in background
    background_tasks.add_task(chembl_fetcher.fetch_all, force_refresh=force_refresh)

    return {
        "status": "fetching",
        "message": "ChEMBL data fetch started in background",
        "force_refresh": force_refresh
    }


@router.get("/training-data")
async def get_training_data(req: Request, organism: Optional[str] = None, limit: int = 1000):
    """
    Get training data for molecular generation models.

    Returns SMILES strings with experimental MIC values from ChEMBL,
    suitable for fine-tuning molecular generation models.
    """
    chembl_fetcher = getattr(req.app.state, "chembl_fetcher", None)
    reference_db = getattr(req.app.state, "reference_db", None)

    training_data = []

    # Get ChEMBL experimental data
    if chembl_fetcher and chembl_fetcher.is_ready:
        chembl_data = chembl_fetcher.get_training_data()

        # Filter by organism if specified
        if organism:
            chembl_data = [
                d for d in chembl_data
                if organism.lower() in d.get("organism", "").lower()
            ]

        training_data.extend(chembl_data)

    # Add reference compound data
    if reference_db and reference_db.is_ready:
        for ref in reference_db.get_all():
            # Build mic_ranges from individual organism MIC tuples
            mic_ranges = {}
            if ref.mic_s_aureus:
                mic_ranges["S. aureus"] = ref.mic_s_aureus
            if ref.mic_e_coli:
                mic_ranges["E. coli"] = ref.mic_e_coli
            if ref.mic_p_aeruginosa:
                mic_ranges["P. aeruginosa"] = ref.mic_p_aeruginosa
            if ref.mic_c_albicans:
                mic_ranges["C. albicans"] = ref.mic_c_albicans

            for org, mic_tuple in mic_ranges.items():
                # Use geometric mean of min/max as representative MIC
                mic_mean = (mic_tuple[0] * mic_tuple[1]) ** 0.5

                if organism and organism.lower() not in org.lower():
                    continue

                training_data.append({
                    "smiles": ref.smiles,
                    "organism": org,
                    "mic_value": mic_mean,
                    "chembl_id": ref.chembl_id,
                    "name": ref.name,
                    "source": "reference"
                })

    # Apply limit
    if len(training_data) > limit:
        training_data = training_data[:limit]

    return {
        "count": len(training_data),
        "data": training_data
    }


@router.get("/smiles-corpus")
async def get_smiles_corpus(req: Request, include_reference: bool = True):
    """
    Get list of SMILES strings for training molecular generation models.

    Returns unique SMILES from both ChEMBL and reference databases.
    """
    chembl_fetcher = getattr(req.app.state, "chembl_fetcher", None)
    reference_db = getattr(req.app.state, "reference_db", None)

    smiles_set = set()

    # Get ChEMBL SMILES
    if chembl_fetcher and chembl_fetcher.is_ready:
        smiles_set.update(chembl_fetcher.get_smiles_list())

    # Get reference SMILES
    if include_reference and reference_db and reference_db.is_ready:
        smiles_set.update(reference_db.get_smiles_list())

    smiles_list = list(smiles_set)

    return {
        "count": len(smiles_list),
        "smiles": smiles_list
    }


# ============================================================================
# REINVENT RL Fine-Tuning Endpoints
# ============================================================================

class RLTrainingRequest(BaseModel):
    """Request model for RL training"""
    constraints: GenerationConstraints = GenerationConstraints()
    weights: GenerationWeights = GenerationWeights()
    max_steps: int = Field(100, ge=1, le=10000)


class RLGenerationRequest(BaseModel):
    """Request model for RL-guided generation"""
    num_molecules: int = Field(100, ge=1, le=10000)
    constraints: GenerationConstraints = GenerationConstraints()
    weights: GenerationWeights = GenerationWeights()
    rl_steps: int = Field(100, ge=1, le=1000)
    batch_size: int = Field(64, ge=1, le=512)


class RLConfigUpdate(BaseModel):
    """Model for updating RL configuration"""
    sigma: Optional[float] = Field(None, ge=1, le=200)
    learning_rate: Optional[float] = Field(None, ge=1e-6, le=1e-2)
    diversity_filter: Optional[bool] = None
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    kl_weight: Optional[float] = Field(None, ge=0.0, le=1.0)


@router.get("/rl/status")
async def get_rl_status(req: Request):
    """
    Get REINVENT RL training status and metrics.

    Returns current training state, configuration, and performance metrics.
    """
    generator = req.app.state.generator
    return generator.rl_training_status


@router.post("/rl/start")
async def start_rl_training(
    request: RLTrainingRequest,
    background_tasks: BackgroundTasks,
    req: Request
):
    """
    Start REINVENT RL fine-tuning loop.

    This trains the agent model to optimize for the specified scoring weights
    while maintaining similarity to the prior model.
    """
    generator = req.app.state.generator

    if not generator.reinvent_trainer:
        raise HTTPException(
            status_code=503,
            detail="REINVENT trainer not initialized. Ensure pretrained model is loaded."
        )

    if generator.rl_training_active:
        raise HTTPException(status_code=409, detail="RL training already in progress")

    if generator.is_running:
        raise HTTPException(status_code=409, detail="Generation already in progress")

    # Validate weights
    total_weight = (
        request.weights.efficacy + request.weights.safety +
        request.weights.environmental + request.weights.sa_score
    )
    if abs(total_weight - 1.0) > 0.01:
        raise HTTPException(
            status_code=400,
            detail=f"Weights must sum to 1.0 (got {total_weight})"
        )

    # Create scoring function
    scoring_function = generator.create_rl_scoring_function(
        request.constraints.model_dump(),
        request.weights.model_dump()
    )

    # Run RL training in background
    background_tasks.add_task(
        generator.start_rl_training,
        scoring_function=scoring_function,
        max_steps=request.max_steps
    )

    return {
        "status": "started",
        "max_steps": request.max_steps,
        "config": {
            "sigma": generator.config.rl_sigma,
            "learning_rate": generator.config.learning_rate,
            "diversity_filter": generator.config.rl_diversity_filter,
        }
    }


@router.post("/rl/stop")
async def stop_rl_training(req: Request):
    """Stop ongoing RL training"""
    generator = req.app.state.generator

    if not generator.rl_training_active:
        raise HTTPException(status_code=400, detail="No RL training in progress")

    await generator.stop_rl_training()

    return {
        "status": "stopped",
        "final_metrics": generator.rl_training_status.get("metrics")
    }


@router.post("/rl/generate")
async def start_rl_generation(
    request: RLGenerationRequest,
    background_tasks: BackgroundTasks,
    req: Request
):
    """
    Start RL-guided molecule generation.

    This combines RL fine-tuning with molecule generation, training the model
    while simultaneously generating and storing valid molecules.
    """
    generator = req.app.state.generator

    if generator.is_running:
        raise HTTPException(status_code=409, detail="Generation already in progress")

    if generator.rl_training_active:
        raise HTTPException(status_code=409, detail="RL training already in progress")

    # Validate weights
    total_weight = (
        request.weights.efficacy + request.weights.safety +
        request.weights.environmental + request.weights.sa_score
    )
    if abs(total_weight - 1.0) > 0.01:
        raise HTTPException(
            status_code=400,
            detail=f"Weights must sum to 1.0 (got {total_weight})"
        )

    # Run RL generation in background
    background_tasks.add_task(
        generator.run_rl_generation,
        num_molecules=request.num_molecules,
        constraints=request.constraints.model_dump(),
        weights=request.weights.model_dump(),
        rl_steps=request.rl_steps,
        batch_size=request.batch_size
    )

    return {
        "status": "started",
        "target_molecules": request.num_molecules,
        "rl_steps": request.rl_steps,
        "rl_available": generator.reinvent_trainer is not None
    }


@router.patch("/rl/config")
async def update_rl_config(config: RLConfigUpdate, req: Request):
    """
    Update RL training configuration.

    Changes take effect on the next training run.
    """
    generator = req.app.state.generator

    if generator.rl_training_active:
        raise HTTPException(
            status_code=409,
            detail="Cannot update config while RL training is active"
        )

    updates = {}

    if config.sigma is not None:
        generator.config.rl_sigma = config.sigma
        updates["sigma"] = config.sigma

    if config.learning_rate is not None:
        generator.config.learning_rate = config.learning_rate
        updates["learning_rate"] = config.learning_rate

    if config.diversity_filter is not None:
        generator.config.rl_diversity_filter = config.diversity_filter
        updates["diversity_filter"] = config.diversity_filter

    if config.similarity_threshold is not None:
        generator.config.rl_similarity_threshold = config.similarity_threshold
        updates["similarity_threshold"] = config.similarity_threshold

    if config.kl_weight is not None:
        generator.config.rl_kl_weight = config.kl_weight
        updates["kl_weight"] = config.kl_weight

    # Re-initialize trainer with new config if it exists
    if updates and generator.reinvent_trainer:
        await generator._setup_reinvent_trainer()

    return {
        "status": "updated",
        "changes": updates,
        "current_config": {
            "sigma": generator.config.rl_sigma,
            "learning_rate": generator.config.learning_rate,
            "diversity_filter": generator.config.rl_diversity_filter,
            "similarity_threshold": generator.config.rl_similarity_threshold,
            "kl_weight": generator.config.rl_kl_weight,
        }
    }


@router.get("/rl/replay-buffer")
async def get_replay_buffer(req: Request, limit: int = 100):
    """
    Get top molecules from the experience replay buffer.

    Returns the highest-scoring molecules seen during RL training.
    """
    generator = req.app.state.generator

    if not generator.reinvent_trainer:
        raise HTTPException(
            status_code=503,
            detail="REINVENT trainer not initialized"
        )

    # Get top entries from replay buffer
    entries = generator.reinvent_trainer.replay_buffer.sample(
        min(limit, len(generator.reinvent_trainer.replay_buffer))
    )

    return {
        "count": len(entries),
        "buffer_size": len(generator.reinvent_trainer.replay_buffer),
        "molecules": [
            {
                "smiles": e.smiles,
                "score": e.score,
                "prior_nll": e.prior_nll,
                "agent_nll": e.agent_nll,
            }
            for e in entries
        ]
    }


@router.post("/rl/reset")
async def reset_rl_trainer(req: Request):
    """
    Reset the REINVENT trainer.

    This clears the replay buffer and diversity filter, and reinitializes
    the agent model from the prior.
    """
    generator = req.app.state.generator

    if generator.rl_training_active:
        raise HTTPException(
            status_code=409,
            detail="Cannot reset while RL training is active"
        )

    if not generator.pretrained_generator or not generator.pretrained_generator.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Pretrained model not available for reset"
        )

    # Re-setup the trainer (creates new agent from prior)
    await generator._setup_reinvent_trainer()

    return {
        "status": "reset",
        "message": "REINVENT trainer reset successfully"
    }
