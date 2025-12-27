"""
Molecule management API routes
CRUD operations for generated molecules
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from pydantic import BaseModel, Field

from database.connection import get_db
from database.models import Molecule
from database import queries

router = APIRouter()


# ============================================================================
# Pydantic Models
# ============================================================================

class MoleculeBase(BaseModel):
    smiles: str
    efficacy_score: float = Field(ge=0, le=100)
    safety_score: float = Field(ge=0, le=100)
    environmental_score: float = Field(ge=0, le=100)
    sa_score: float = Field(ge=0, le=100)


class MoleculeCreate(MoleculeBase):
    pass


class MoleculeResponse(MoleculeBase):
    id: int
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    chain_length: Optional[int] = None
    is_valid_quat: bool = True
    is_pareto: bool = False
    is_starred: bool = False
    created_at: str
    
    class Config:
        from_attributes = True


class MoleculeListResponse(BaseModel):
    molecules: list[MoleculeResponse]
    total: int
    offset: int
    limit: int


class MoleculeDetailResponse(MoleculeResponse):
    """Extended molecule info with properties and similar molecules"""
    properties: dict = {}
    similar_molecules: list[int] = []
    generation_info: dict = {}


class MoleculeUpdateRequest(BaseModel):
    is_starred: Optional[bool] = None


# ============================================================================
# Routes
# ============================================================================

@router.get("", response_model=MoleculeListResponse)
async def list_molecules(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    pareto_only: bool = Query(False),
    starred_only: bool = Query(False),
    min_efficacy: Optional[float] = Query(None, ge=0, le=100),
    min_safety: Optional[float] = Query(None, ge=0, le=100),
    min_environmental: Optional[float] = Query(None, ge=0, le=100),
    min_sa: Optional[float] = Query(None, ge=0, le=100),
    sort_by: str = Query("created_at", regex="^(created_at|efficacy_score|safety_score|environmental_score|sa_score)$"),
    sort_desc: bool = Query(True),
    db: AsyncSession = Depends(get_db)
):
    """
    List molecules with filtering and pagination
    
    Supports filtering by:
    - Pareto optimality
    - Starred status
    - Minimum scores for each objective
    
    Supports sorting by any score column
    """
    filters = {
        "pareto_only": pareto_only,
        "starred_only": starred_only,
        "min_efficacy": min_efficacy,
        "min_safety": min_safety,
        "min_environmental": min_environmental,
        "min_sa": min_sa,
    }
    
    molecules, total = await queries.get_molecules(
        db,
        limit=limit,
        offset=offset,
        filters=filters,
        sort_by=sort_by,
        sort_desc=sort_desc
    )
    
    return MoleculeListResponse(
        molecules=[MoleculeResponse.model_validate(m) for m in molecules],
        total=total,
        offset=offset,
        limit=limit
    )


@router.get("/{molecule_id}", response_model=MoleculeDetailResponse)
async def get_molecule(
    molecule_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get detailed information about a specific molecule"""
    molecule = await queries.get_molecule_by_id(db, molecule_id)
    
    if not molecule:
        raise HTTPException(status_code=404, detail="Molecule not found")
    
    # Get additional properties
    properties = await queries.get_molecule_properties(db, molecule_id)
    similar = await queries.get_similar_molecules(db, molecule_id, limit=10)
    generation_info = await queries.get_generation_info(db, molecule_id)
    
    response = MoleculeDetailResponse.model_validate(molecule)
    response.properties = properties
    response.similar_molecules = [m.id for m in similar]
    response.generation_info = generation_info
    
    return response


@router.patch("/{molecule_id}", response_model=MoleculeResponse)
async def update_molecule(
    molecule_id: int,
    update: MoleculeUpdateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Update molecule metadata (e.g., starred status)"""
    molecule = await queries.get_molecule_by_id(db, molecule_id)
    
    if not molecule:
        raise HTTPException(status_code=404, detail="Molecule not found")
    
    if update.is_starred is not None:
        molecule = await queries.update_molecule_starred(
            db, molecule_id, update.is_starred
        )
    
    return MoleculeResponse.model_validate(molecule)


@router.delete("/{molecule_id}")
async def delete_molecule(
    molecule_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a molecule from the database"""
    success = await queries.delete_molecule(db, molecule_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Molecule not found")
    
    return {"status": "deleted", "id": molecule_id}


@router.get("/search/smiles")
async def search_by_smiles(
    smiles: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db)
):
    """Search for a molecule by exact SMILES match"""
    molecule = await queries.get_molecule_by_smiles(db, smiles)
    
    if not molecule:
        raise HTTPException(status_code=404, detail="Molecule not found")
    
    return MoleculeResponse.model_validate(molecule)


@router.get("/stats/summary")
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Get summary statistics about generated molecules"""
    stats = await queries.get_molecule_stats(db)
    return stats


@router.post("/bulk/star")
async def bulk_star(
    molecule_ids: list[int],
    starred: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """Star or unstar multiple molecules at once"""
    count = await queries.bulk_update_starred(db, molecule_ids, starred)
    return {"updated": count}


@router.delete("/bulk/delete")
async def bulk_delete(
    molecule_ids: list[int],
    db: AsyncSession = Depends(get_db)
):
    """Delete multiple molecules at once"""
    count = await queries.bulk_delete_molecules(db, molecule_ids)
    return {"deleted": count}
