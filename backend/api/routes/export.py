"""
Export API routes
Export molecules in various formats (CSV, SDF, PDF)
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal
from sqlalchemy.ext.asyncio import AsyncSession
import tempfile
import os

from database.connection import get_db
from database import queries
from export.csv_export import export_to_csv
from export.sdf_export import export_to_sdf
from export.pdf_export import export_to_pdf

router = APIRouter()


# ============================================================================
# Pydantic Models
# ============================================================================

class ExportRequest(BaseModel):
    """Request to export molecules"""
    molecule_ids: Optional[list[int]] = None  # None = export all filtered
    format: Literal["csv", "sdf", "pdf"] = "csv"
    include_structures: bool = True
    include_properties: bool = True
    include_scores: bool = True
    
    # Filters (applied if molecule_ids is None)
    pareto_only: bool = False
    starred_only: bool = False
    min_efficacy: Optional[float] = None
    min_safety: Optional[float] = None
    min_environmental: Optional[float] = None


class ExportColumn(BaseModel):
    """Column configuration for CSV export"""
    name: str
    include: bool = True


class CsvExportOptions(BaseModel):
    """Options specific to CSV export"""
    delimiter: str = ","
    include_header: bool = True
    columns: Optional[list[str]] = None  # None = all columns


class PdfExportOptions(BaseModel):
    """Options specific to PDF export"""
    title: str = "Quat Generator Pro - Export Report"
    include_summary: bool = True
    include_pareto_plot: bool = True
    structures_per_page: int = 6
    page_size: Literal["letter", "a4"] = "letter"


# ============================================================================
# Routes
# ============================================================================

@router.post("/csv")
async def export_csv(
    request: ExportRequest,
    options: Optional[CsvExportOptions] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Export molecules to CSV format
    
    Returns a downloadable CSV file with SMILES, scores, and properties.
    """
    if options is None:
        options = CsvExportOptions()
    
    # Get molecules
    molecules = await _get_molecules_for_export(db, request)
    
    if not molecules:
        raise HTTPException(status_code=404, detail="No molecules to export")
    
    # Generate CSV
    csv_content = await export_to_csv(
        molecules,
        delimiter=options.delimiter,
        include_header=options.include_header,
        columns=options.columns,
        include_properties=request.include_properties,
        include_scores=request.include_scores
    )
    
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=quat_molecules.csv"
        }
    )


@router.post("/sdf")
async def export_sdf(
    request: ExportRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Export molecules to SDF (Structure-Data File) format
    
    Standard chemical format readable by most chemistry software.
    Includes 2D coordinates and all properties as data fields.
    """
    molecules = await _get_molecules_for_export(db, request)
    
    if not molecules:
        raise HTTPException(status_code=404, detail="No molecules to export")
    
    # Generate SDF
    sdf_content = await export_to_sdf(
        molecules,
        include_properties=request.include_properties,
        include_scores=request.include_scores
    )
    
    return StreamingResponse(
        iter([sdf_content]),
        media_type="chemical/x-mdl-sdfile",
        headers={
            "Content-Disposition": "attachment; filename=quat_molecules.sdf"
        }
    )


@router.post("/pdf")
async def export_pdf(
    request: ExportRequest,
    options: Optional[PdfExportOptions] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Export molecules to PDF report
    
    Generates a professional report with:
    - Summary statistics
    - Pareto frontier visualization
    - 2D structure images
    - Property tables
    """
    if options is None:
        options = PdfExportOptions()
    
    molecules = await _get_molecules_for_export(db, request)
    
    if not molecules:
        raise HTTPException(status_code=404, detail="No molecules to export")
    
    # Generate PDF to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        await export_to_pdf(
            molecules,
            output_path=tmp.name,
            title=options.title,
            include_summary=options.include_summary,
            include_pareto_plot=options.include_pareto_plot,
            structures_per_page=options.structures_per_page,
            page_size=options.page_size,
            include_properties=request.include_properties,
            include_scores=request.include_scores
        )
        
        return FileResponse(
            tmp.name,
            media_type="application/pdf",
            filename="quat_molecules_report.pdf",
            background=None  # Don't delete file until response is sent
        )


@router.get("/formats")
async def list_formats():
    """List available export formats with descriptions"""
    return {
        "formats": [
            {
                "id": "csv",
                "name": "CSV (Comma-Separated Values)",
                "extension": ".csv",
                "description": "Tabular data format, opens in Excel/Sheets",
                "supports_structures": False
            },
            {
                "id": "sdf",
                "name": "SDF (Structure-Data File)",
                "extension": ".sdf",
                "description": "Standard chemistry format with 2D structures",
                "supports_structures": True
            },
            {
                "id": "pdf",
                "name": "PDF Report",
                "extension": ".pdf",
                "description": "Professional report with visualizations",
                "supports_structures": True
            }
        ]
    }


@router.get("/preview/{format}")
async def preview_export(
    format: Literal["csv", "sdf", "pdf"],
    molecule_ids: str = Query(..., description="Comma-separated molecule IDs"),
    db: AsyncSession = Depends(get_db)
):
    """
    Preview export content (first few rows/molecules)
    
    Useful for verifying export settings before downloading.
    """
    ids = [int(x.strip()) for x in molecule_ids.split(",")][:5]  # Max 5 for preview
    
    request = ExportRequest(molecule_ids=ids, format=format)
    molecules = await _get_molecules_for_export(db, request)
    
    if format == "csv":
        content = await export_to_csv(molecules, columns=None)
        return {"preview": content[:2000], "truncated": len(content) > 2000}
    elif format == "sdf":
        content = await export_to_sdf(molecules)
        return {"preview": content[:2000], "truncated": len(content) > 2000}
    else:
        return {"preview": f"PDF preview for {len(molecules)} molecules", "truncated": False}


# ============================================================================
# Helper Functions
# ============================================================================

async def _get_molecules_for_export(
    db: AsyncSession,
    request: ExportRequest
) -> list:
    """Get molecules based on export request filters"""
    
    if request.molecule_ids:
        # Get specific molecules by ID
        molecules = await queries.get_molecules_by_ids(db, request.molecule_ids)
    else:
        # Apply filters
        filters = {
            "pareto_only": request.pareto_only,
            "starred_only": request.starred_only,
            "min_efficacy": request.min_efficacy,
            "min_safety": request.min_safety,
            "min_environmental": request.min_environmental,
        }
        molecules, _ = await queries.get_molecules(
            db,
            limit=10000,  # Max export size
            offset=0,
            filters=filters
        )
    
    return molecules
