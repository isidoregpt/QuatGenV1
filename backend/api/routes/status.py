"""
Status API routes
System status, GPU info, and diagnostics
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional
import platform
import psutil
import torch
import os

router = APIRouter()


# ============================================================================
# Pydantic Models
# ============================================================================

class SystemInfo(BaseModel):
    """System information"""
    platform: str
    python_version: str
    cpu_count: int
    cpu_percent: float
    memory_total_gb: float
    memory_available_gb: float
    memory_percent: float


class GpuInfo(BaseModel):
    """GPU information"""
    available: bool
    device_count: int
    devices: list[dict]
    cuda_version: Optional[str]


class ModelInfo(BaseModel):
    """Model status information"""
    generator_loaded: bool
    generator_device: str
    scoring_models_loaded: bool
    vocab_size: int


class FullStatus(BaseModel):
    """Complete system status"""
    system: SystemInfo
    gpu: GpuInfo
    models: ModelInfo
    database: dict
    version: str


# ============================================================================
# Routes
# ============================================================================

@router.get("/system", response_model=SystemInfo)
async def get_system_info():
    """Get system information (CPU, memory)"""
    memory = psutil.virtual_memory()
    
    return SystemInfo(
        platform=platform.platform(),
        python_version=platform.python_version(),
        cpu_count=psutil.cpu_count(),
        cpu_percent=psutil.cpu_percent(interval=0.1),
        memory_total_gb=memory.total / (1024**3),
        memory_available_gb=memory.available / (1024**3),
        memory_percent=memory.percent
    )


@router.get("/gpu", response_model=GpuInfo)
async def get_gpu_info():
    """Get GPU information"""
    if not torch.cuda.is_available():
        return GpuInfo(
            available=False,
            device_count=0,
            devices=[],
            cuda_version=None
        )
    
    devices = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_total = props.total_memory / (1024**3)
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
        memory_cached = torch.cuda.memory_reserved(i) / (1024**3)
        
        devices.append({
            "index": i,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "memory_total_gb": round(memory_total, 2),
            "memory_allocated_gb": round(memory_allocated, 2),
            "memory_cached_gb": round(memory_cached, 2),
            "memory_free_gb": round(memory_total - memory_allocated, 2),
            "multi_processor_count": props.multi_processor_count
        })
    
    return GpuInfo(
        available=True,
        device_count=torch.cuda.device_count(),
        devices=devices,
        cuda_version=torch.version.cuda
    )


@router.get("/models", response_model=ModelInfo)
async def get_model_info(req: Request):
    """Get loaded model information"""
    generator = req.app.state.generator
    
    return ModelInfo(
        generator_loaded=generator.is_ready if generator else False,
        generator_device=str(generator.device) if generator else "none",
        scoring_models_loaded=generator.scoring_ready if generator else False,
        vocab_size=generator.vocab_size if generator else 0
    )


@router.get("/full", response_model=FullStatus)
async def get_full_status(req: Request):
    """Get complete system status"""
    system_info = await get_system_info()
    gpu_info = await get_gpu_info()
    model_info = await get_model_info(req)
    
    # Database info
    from database.connection import get_db_stats
    db_stats = await get_db_stats()
    
    return FullStatus(
        system=system_info,
        gpu=gpu_info,
        models=model_info,
        database=db_stats,
        version="0.1.0"
    )


@router.get("/benchmark")
async def run_benchmark(req: Request):
    """
    Run a quick performance benchmark
    
    Tests:
    - SMILES tokenization speed
    - Model inference speed
    - Scoring pipeline speed
    """
    generator = req.app.state.generator
    
    if not generator or not generator.is_ready:
        return {"error": "Generator not ready"}
    
    import time
    results = {}
    
    # Test tokenization
    test_smiles = "CCCCCCCCCCCC[N+](C)(C)Cc1ccccc1.[Cl-]"
    
    start = time.perf_counter()
    for _ in range(1000):
        generator.tokenize(test_smiles)
    results["tokenization_per_sec"] = 1000 / (time.perf_counter() - start)
    
    # Test generation (single molecule)
    start = time.perf_counter()
    for _ in range(10):
        generator.generate_one()
    results["generation_per_sec"] = 10 / (time.perf_counter() - start)
    
    # Test scoring
    start = time.perf_counter()
    for _ in range(100):
        generator.score_molecule(test_smiles)
    results["scoring_per_sec"] = 100 / (time.perf_counter() - start)
    
    # Estimate throughput
    results["estimated_molecules_per_hour"] = min(
        results["generation_per_sec"],
        results["scoring_per_sec"]
    ) * 3600
    
    return results


@router.post("/clear-gpu-cache")
async def clear_gpu_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return {"status": "cleared"}
    return {"status": "no GPU available"}


@router.get("/logs")
async def get_recent_logs(lines: int = 100):
    """Get recent log entries"""
    log_file = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "app.log")
    
    if not os.path.exists(log_file):
        return {"logs": [], "message": "No log file found"}
    
    with open(log_file, "r") as f:
        all_lines = f.readlines()
        recent = all_lines[-lines:]
    
    return {"logs": recent, "total_lines": len(all_lines)}
