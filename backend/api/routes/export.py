"""Export API routes"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Literal
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from database import queries
from export.csv_export import export_to_csv
from export.sdf_export import export_to_sdf

router = APIRouter()


class ExportRequest(BaseModel):
    molecule_ids: Optional[list[int]] = None
    format: Literal["csv", "sdf", "pdf"] = "csv"
    include_structures: bool = True
    include_properties: bool = True
    include_scores: bool = True
    pareto_only: bool = False
    starred_only: bool = False


@router.post("/csv")
async def export_csv(request: ExportRequest, db: AsyncSession = Depends(get_db)):
    molecules = await _get_molecules_for_export(db, request)
    if not molecules:
        raise HTTPException(status_code=404, detail="No molecules to export")
    csv_content = await export_to_csv(molecules, include_properties=request.include_properties, include_scores=request.include_scores)
    return StreamingResponse(iter([csv_content]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=quat_molecules.csv"})


@router.post("/sdf")
async def export_sdf(request: ExportRequest, db: AsyncSession = Depends(get_db)):
    molecules = await _get_molecules_for_export(db, request)
    if not molecules:
        raise HTTPException(status_code=404, detail="No molecules to export")
    sdf_content = await export_to_sdf(molecules, include_properties=request.include_properties, include_scores=request.include_scores)
    return StreamingResponse(iter([sdf_content]), media_type="chemical/x-mdl-sdfile", headers={"Content-Disposition": "attachment; filename=quat_molecules.sdf"})


@router.get("/formats")
async def list_formats():
    return {"formats": [
        {"id": "csv", "name": "CSV", "extension": ".csv", "description": "Tabular data format"},
        {"id": "sdf", "name": "SDF", "extension": ".sdf", "description": "Standard chemistry format"},
        {"id": "pdf", "name": "PDF", "extension": ".pdf", "description": "Report with visualizations"}
    ]}


async def _get_molecules_for_export(db: AsyncSession, request: ExportRequest) -> list:
    if request.molecule_ids:
        return await queries.get_molecules_by_ids(db, request.molecule_ids)
    filters = {"pareto_only": request.pareto_only, "starred_only": request.starred_only}
    molecules, _ = await queries.get_molecules(db, limit=10000, offset=0, filters=filters)
    return molecules
