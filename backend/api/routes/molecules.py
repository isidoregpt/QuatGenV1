"""Molecule management API routes"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from pydantic import BaseModel, Field

from database.connection import get_db
from database import queries

router = APIRouter()


class MoleculeBase(BaseModel):
    smiles: str
    efficacy_score: float = Field(ge=0, le=100)
    safety_score: float = Field(ge=0, le=100)
    environmental_score: float = Field(ge=0, le=100)
    sa_score: float = Field(ge=0, le=100)


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


class MoleculeUpdateRequest(BaseModel):
    is_starred: Optional[bool] = None


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
    sort_by: str = Query("created_at"),
    sort_desc: bool = Query(True),
    db: AsyncSession = Depends(get_db)
):
    filters = {"pareto_only": pareto_only, "starred_only": starred_only,
               "min_efficacy": min_efficacy, "min_safety": min_safety,
               "min_environmental": min_environmental, "min_sa": min_sa}
    molecules, total = await queries.get_molecules(db, limit=limit, offset=offset, filters=filters, sort_by=sort_by, sort_desc=sort_desc)
    return MoleculeListResponse(molecules=[MoleculeResponse.model_validate(m) for m in molecules], total=total, offset=offset, limit=limit)


@router.get("/{molecule_id}")
async def get_molecule(molecule_id: int, db: AsyncSession = Depends(get_db)):
    molecule = await queries.get_molecule_by_id(db, molecule_id)
    if not molecule:
        raise HTTPException(status_code=404, detail="Molecule not found")
    properties = await queries.get_molecule_properties(db, molecule_id)
    similar = await queries.get_similar_molecules(db, molecule_id, limit=10)
    generation_info = await queries.get_generation_info(db, molecule_id)
    response = MoleculeResponse.model_validate(molecule)
    return {"molecule": response, "properties": properties, "similar_molecules": [m.id for m in similar], "generation_info": generation_info}


@router.patch("/{molecule_id}", response_model=MoleculeResponse)
async def update_molecule(molecule_id: int, update: MoleculeUpdateRequest, db: AsyncSession = Depends(get_db)):
    molecule = await queries.get_molecule_by_id(db, molecule_id)
    if not molecule:
        raise HTTPException(status_code=404, detail="Molecule not found")
    if update.is_starred is not None:
        molecule = await queries.update_molecule_starred(db, molecule_id, update.is_starred)
    return MoleculeResponse.model_validate(molecule)


@router.delete("/{molecule_id}")
async def delete_molecule(molecule_id: int, db: AsyncSession = Depends(get_db)):
    success = await queries.delete_molecule(db, molecule_id)
    if not success:
        raise HTTPException(status_code=404, detail="Molecule not found")
    return {"status": "deleted", "id": molecule_id}


@router.get("/stats/summary")
async def get_stats(db: AsyncSession = Depends(get_db)):
    return await queries.get_molecule_stats(db)


@router.post("/bulk/star")
async def bulk_star(molecule_ids: list[int], starred: bool = True, db: AsyncSession = Depends(get_db)):
    count = await queries.bulk_update_starred(db, molecule_ids, starred)
    return {"updated": count}
