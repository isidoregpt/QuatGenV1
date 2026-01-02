"""Status API routes"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import platform
import psutil
import torch
import os

router = APIRouter()


class ChEMBLStatus(BaseModel):
    is_ready: bool
    compound_count: int
    compounds_with_mic: int
    organisms: List[str]
    cache_available: bool


class ReferenceCompound(BaseModel):
    name: str
    chembl_id: str
    smiles: str
    category: str
    molecular_weight: float
    applications: List[str]
    mic_ranges: Dict[str, Dict[str, float]]
    ld50_oral_rat: Optional[float]
    regulatory_status: str


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


@router.get("/chembl", response_model=ChEMBLStatus)
async def get_chembl_status(req: Request):
    """Get ChEMBL data fetcher status and statistics"""
    chembl_fetcher = getattr(req.app.state, "chembl_fetcher", None)

    if not chembl_fetcher:
        return ChEMBLStatus(
            is_ready=False,
            compound_count=0,
            compounds_with_mic=0,
            organisms=[],
            cache_available=False
        )

    cache_file = os.path.join(chembl_fetcher.cache_dir, "quat_compounds.json")

    return ChEMBLStatus(
        is_ready=chembl_fetcher.is_ready,
        compound_count=chembl_fetcher.compound_count,
        compounds_with_mic=chembl_fetcher.compounds_with_mic_count,
        organisms=chembl_fetcher.get_organisms(),
        cache_available=os.path.exists(cache_file)
    )


@router.get("/references", response_model=List[ReferenceCompound])
async def get_reference_compounds(req: Request):
    """Get curated reference quaternary ammonium compounds"""
    reference_db = getattr(req.app.state, "reference_db", None)

    if not reference_db:
        return []

    compounds = []
    for ref in reference_db.get_all():
        compounds.append(ReferenceCompound(
            name=ref.name,
            chembl_id=ref.chembl_id,
            smiles=ref.smiles,
            category=ref.category,
            molecular_weight=ref.molecular_weight,
            applications=ref.applications,
            mic_ranges=ref.mic_ranges,
            ld50_oral_rat=ref.ld50_oral_rat,
            regulatory_status=ref.regulatory_status
        ))

    return compounds


@router.get("/references/{chembl_id}")
async def get_reference_compound(chembl_id: str, req: Request):
    """Get a specific reference compound by ChEMBL ID"""
    reference_db = getattr(req.app.state, "reference_db", None)

    if not reference_db:
        return {"error": "Reference database not initialized"}

    ref = reference_db.get(chembl_id)
    if not ref:
        return {"error": f"Compound {chembl_id} not found"}

    return ref.to_dict()
