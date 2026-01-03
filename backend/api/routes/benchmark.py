"""
Benchmark API routes
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel, Field

from database.connection import get_db
from database import queries
from benchmark.comparator import BenchmarkComparator
from benchmark.report import BenchmarkReport

router = APIRouter()


class BenchmarkRequest(BaseModel):
    smiles: str
    predicted_scores: Optional[dict] = None


class BatchBenchmarkRequest(BaseModel):
    molecule_ids: Optional[List[int]] = None
    top_n: int = Field(default=20, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=100.0)


class ReportRequest(BaseModel):
    molecule_ids: Optional[List[int]] = None


@router.get("/status")
async def get_benchmark_status(req: Request):
    """Get benchmark service status"""
    reference_db = getattr(req.app.state, 'reference_db', None)

    comparator = BenchmarkComparator(reference_db)

    return {
        "available": comparator.is_ready,
        "reference_compounds": len(reference_db.get_all()) if reference_db else 0,
        "features": [
            "single_molecule_benchmark",
            "batch_benchmark",
            "report_generation",
            "property_comparison",
            "scaffold_classification"
        ]
    }


@router.post("/molecule")
async def benchmark_single_molecule(
    request: BenchmarkRequest,
    req: Request
):
    """Benchmark a single molecule against references"""
    reference_db = getattr(req.app.state, 'reference_db', None)

    if not reference_db:
        raise HTTPException(status_code=503, detail="Reference database not available")

    comparator = BenchmarkComparator(reference_db)

    if not comparator.is_ready:
        raise HTTPException(status_code=503, detail="Benchmark comparator not ready")

    result = comparator.benchmark_molecule(
        request.smiles,
        request.predicted_scores
    )

    return {
        "smiles": result.smiles,
        "overall_score": result.overall_score,
        "recommendation": result.recommendation,
        "confidence": result.confidence,
        "scaffold_type": result.scaffold_type,
        "structural_novelty": result.structural_novelty,
        "closest_references": result.closest_references,
        "property_comparisons": [
            {
                "property": pc.property_name,
                "generated": pc.generated_value,
                "reference": pc.reference_value,
                "outcome": pc.outcome.value,
                "interpretation": pc.interpretation
            }
            for pc in result.property_comparisons
        ],
        "advantages": result.predicted_advantages,
        "disadvantages": result.predicted_disadvantages,
        "properties_better": result.properties_better,
        "properties_similar": result.properties_similar,
        "properties_worse": result.properties_worse
    }


@router.post("/batch")
async def benchmark_batch(
    request: BatchBenchmarkRequest,
    db: AsyncSession = Depends(get_db),
    req: Request = None
):
    """Benchmark multiple molecules from database"""
    reference_db = getattr(req.app.state, 'reference_db', None)

    if not reference_db:
        raise HTTPException(status_code=503, detail="Reference database not available")

    comparator = BenchmarkComparator(reference_db)

    if not comparator.is_ready:
        raise HTTPException(status_code=503, detail="Benchmark comparator not ready")

    # Get molecules
    if request.molecule_ids:
        molecules = []
        for mol_id in request.molecule_ids:
            mol = await queries.get_molecule_by_id(db, mol_id)
            if mol:
                molecules.append(mol)
    else:
        molecules = await queries.get_all_molecules(db, limit=500)

    if not molecules:
        return {"results": [], "count": 0, "molecules_analyzed": 0}

    # Prepare for benchmarking
    mol_tuples = [
        (m.smiles,
         {"efficacy_score": m.efficacy_score, "safety_score": m.safety_score,
          "sa_score": m.sa_score, "environmental_score": m.environmental_score},
         m.id)
        for m in molecules
    ]

    # Benchmark
    results = comparator.benchmark_batch(mol_tuples, request.top_n)

    # Filter by minimum score
    results = [r for r in results if r.overall_score >= request.min_score]

    return {
        "count": len(results),
        "molecules_analyzed": len(mol_tuples),
        "results": [
            {
                "molecule_id": r.molecule_id,
                "smiles": r.smiles,
                "overall_score": r.overall_score,
                "recommendation": r.recommendation,
                "scaffold_type": r.scaffold_type,
                "structural_novelty": r.structural_novelty,
                "closest_reference": r.closest_references[0]["name"] if r.closest_references else None,
                "advantages_count": len(r.predicted_advantages),
                "disadvantages_count": len(r.predicted_disadvantages)
            }
            for r in results
        ]
    }


@router.post("/report")
async def generate_benchmark_report(
    request: ReportRequest = None,
    db: AsyncSession = Depends(get_db),
    req: Request = None
):
    """Generate a comprehensive benchmark report"""
    reference_db = getattr(req.app.state, 'reference_db', None)

    if not reference_db:
        raise HTTPException(status_code=503, detail="Reference database not available")

    comparator = BenchmarkComparator(reference_db)

    if not comparator.is_ready:
        raise HTTPException(status_code=503, detail="Benchmark comparator not ready")

    # Get molecules
    molecule_ids = request.molecule_ids if request else None
    if molecule_ids:
        molecules = []
        for mol_id in molecule_ids:
            mol = await queries.get_molecule_by_id(db, mol_id)
            if mol:
                molecules.append(mol)
    else:
        molecules = await queries.get_all_molecules(db, limit=200)

    if not molecules:
        raise HTTPException(status_code=404, detail="No molecules found")

    # Benchmark all molecules
    results = []
    for mol in molecules:
        scores = {
            "efficacy_score": mol.efficacy_score,
            "safety_score": mol.safety_score,
            "sa_score": mol.sa_score,
            "environmental_score": mol.environmental_score
        }
        result = comparator.benchmark_molecule(mol.smiles, scores, mol.id)
        results.append(result)

    # Generate report
    report = BenchmarkReport.generate(results, reference_db)

    return report.to_dict()


@router.get("/references")
async def get_reference_compounds(req: Request):
    """Get all reference compounds used for benchmarking"""
    reference_db = getattr(req.app.state, 'reference_db', None)

    if not reference_db:
        raise HTTPException(status_code=503, detail="Reference database not available")

    compounds = reference_db.get_all()

    return {
        "count": len(compounds),
        "compounds": [
            {
                "name": c.name,
                "smiles": c.smiles,
                "chembl_id": c.chembl_id,
                "mic_s_aureus": c.mic_s_aureus,
                "mic_e_coli": c.mic_e_coli,
                "mic_p_aeruginosa": c.mic_p_aeruginosa,
                "mic_c_albicans": c.mic_c_albicans,
                "ld50_oral_rat": c.ld50_oral_rat,
                "applications": c.applications
            }
            for c in compounds
        ]
    }


@router.get("/criteria")
async def get_benchmark_criteria():
    """Get benchmark comparison criteria and thresholds"""
    return {
        "property_thresholds": BenchmarkComparator.SIMILARITY_THRESHOLDS,
        "optimization_directions": {
            k: v for k, v in BenchmarkComparator.OPTIMIZATION_DIRECTION.items()
        },
        "score_interpretation": {
            "80-100": "Highly promising - prioritize for validation",
            "65-79": "Good candidate - comparable to established quats",
            "50-64": "Moderate - may need optimization",
            "35-49": "Below average - significant improvements needed",
            "0-34": "Poor candidate - consider alternatives"
        },
        "scaffold_types": [
            "pyridinium",
            "imidazolium",
            "benzylammonium",
            "tetraalkylammonium",
            "aromatic_quat",
            "aliphatic_quat"
        ]
    }


@router.get("/molecule/{molecule_id}")
async def benchmark_molecule_by_id(
    molecule_id: int,
    db: AsyncSession = Depends(get_db),
    req: Request = None
):
    """Benchmark a specific molecule from the database by ID"""
    reference_db = getattr(req.app.state, 'reference_db', None)

    if not reference_db:
        raise HTTPException(status_code=503, detail="Reference database not available")

    comparator = BenchmarkComparator(reference_db)

    if not comparator.is_ready:
        raise HTTPException(status_code=503, detail="Benchmark comparator not ready")

    molecule = await queries.get_molecule_by_id(db, molecule_id)
    if not molecule:
        raise HTTPException(status_code=404, detail="Molecule not found")

    predicted_scores = {
        "efficacy_score": molecule.efficacy_score,
        "safety_score": molecule.safety_score,
        "sa_score": molecule.sa_score,
        "environmental_score": molecule.environmental_score
    }

    result = comparator.benchmark_molecule(
        molecule.smiles,
        predicted_scores,
        molecule.id
    )

    return {
        "molecule_id": result.molecule_id,
        "smiles": result.smiles,
        "overall_score": result.overall_score,
        "recommendation": result.recommendation,
        "confidence": result.confidence,
        "scaffold_type": result.scaffold_type,
        "structural_novelty": result.structural_novelty,
        "closest_references": result.closest_references,
        "property_comparisons": [
            {
                "property": pc.property_name,
                "generated": pc.generated_value,
                "reference": pc.reference_value,
                "outcome": pc.outcome.value,
                "interpretation": pc.interpretation
            }
            for pc in result.property_comparisons
        ],
        "advantages": result.predicted_advantages,
        "disadvantages": result.predicted_disadvantages,
        "properties_better": result.properties_better,
        "properties_similar": result.properties_similar,
        "properties_worse": result.properties_worse
    }
