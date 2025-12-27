"""Status API routes"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional
import platform
import psutil
import torch
import os

router = APIRouter()


class SystemInfo(BaseModel):
    platform: str
    python_version: str
    cpu_count: int
    cpu_percent: float
    memory_total_gb: float
    memory_available_gb: float


class GpuInfo(BaseModel):
    available: bool
    device_count: int
    devices: list[dict]
    cuda_version: Optional[str]


@router.get("/system", response_model=SystemInfo)
async def get_system_info():
    memory = psutil.virtual_memory()
    return SystemInfo(platform=platform.platform(), python_version=platform.python_version(),
                     cpu_count=psutil.cpu_count(), cpu_percent=psutil.cpu_percent(interval=0.1),
                     memory_total_gb=memory.total / (1024**3), memory_available_gb=memory.available / (1024**3))


@router.get("/gpu", response_model=GpuInfo)
async def get_gpu_info():
    if not torch.cuda.is_available():
        return GpuInfo(available=False, device_count=0, devices=[], cuda_version=None)
    devices = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_total = props.total_memory / (1024**3)
        devices.append({"index": i, "name": props.name, "memory_total_gb": round(memory_total, 2),
                       "memory_free_gb": round(memory_total - torch.cuda.memory_allocated(i) / (1024**3), 2)})
    return GpuInfo(available=True, device_count=torch.cuda.device_count(), devices=devices, cuda_version=torch.version.cuda)


@router.get("/models")
async def get_model_info(req: Request):
    generator = req.app.state.generator
    return {"generator_loaded": generator.is_ready if generator else False, "generator_device": str(generator.device) if generator else "none",
            "scoring_models_loaded": generator.scoring_ready if generator else False, "vocab_size": generator.vocab_size if generator else 0}


@router.get("/full")
async def get_full_status(req: Request):
    from database.connection import get_db_stats
    return {"system": await get_system_info(), "gpu": await get_gpu_info(), "models": await get_model_info(req),
            "database": await get_db_stats(), "version": "0.1.0"}
