"""Molecule management API routes"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
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


class EmbeddingResponse(BaseModel):
    molecule_id: int
    smiles: str
    embedding: List[float]
    embedding_dim: int


class SimilarityRequest(BaseModel):
    smiles1: str
    smiles2: str


class SimilarityResponse(BaseModel):
    smiles1: str
    smiles2: str
    similarity: float


@router.get("/{molecule_id}/embedding", response_model=EmbeddingResponse)
async def get_molecule_embedding(
    molecule_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Get the molecular embedding for a specific molecule."""
    molecule = await queries.get_molecule_by_id(db, molecule_id)
    if not molecule:
        raise HTTPException(status_code=404, detail="Molecule not found")

    # Get the scoring pipeline from app state
    if not hasattr(request.app.state, 'generator') or request.app.state.generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")

    pipeline = request.app.state.generator.scoring
    if not pipeline or not pipeline.encoder_ready:
        raise HTTPException(status_code=503, detail="Molecular encoder not available")

    embedding = await pipeline.get_embedding(molecule.smiles)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Failed to compute embedding")

    return EmbeddingResponse(
        molecule_id=molecule_id,
        smiles=molecule.smiles,
        embedding=embedding.tolist(),
        embedding_dim=len(embedding)
    )


@router.post("/embedding/compute")
async def compute_embedding(
    smiles: str,
    request: Request
):
    """Compute embedding for an arbitrary SMILES string."""
    if not hasattr(request.app.state, 'generator') or request.app.state.generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")

    pipeline = request.app.state.generator.scoring
    if not pipeline or not pipeline.encoder_ready:
        raise HTTPException(status_code=503, detail="Molecular encoder not available")

    embedding = await pipeline.get_embedding(smiles)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Failed to compute embedding")

    return {
        "smiles": smiles,
        "embedding": embedding.tolist(),
        "embedding_dim": len(embedding)
    }


@router.post("/embedding/similarity", response_model=SimilarityResponse)
async def compute_similarity(
    similarity_request: SimilarityRequest,
    request: Request
):
    """Compute similarity between two molecules based on their embeddings."""
    if not hasattr(request.app.state, 'generator') or request.app.state.generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")

    pipeline = request.app.state.generator.scoring
    if not pipeline or not pipeline.encoder_ready:
        raise HTTPException(status_code=503, detail="Molecular encoder not available")

    similarity = await pipeline.get_similarity(
        similarity_request.smiles1,
        similarity_request.smiles2
    )
    if similarity is None:
        raise HTTPException(status_code=500, detail="Failed to compute similarity")

    return SimilarityResponse(
        smiles1=similarity_request.smiles1,
        smiles2=similarity_request.smiles2,
        similarity=similarity
    )


# ADMET Prediction Endpoints

class ADMETPredictRequest(BaseModel):
    smiles: str
    properties: Optional[List[str]] = None  # None = predict all


@router.get("/{molecule_id}/admet")
async def get_molecule_admet(
    molecule_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Get ADMET predictions for a specific molecule."""
    molecule = await queries.get_molecule_by_id(db, molecule_id)
    if not molecule:
        raise HTTPException(status_code=404, detail="Molecule not found")

    # Get the scoring pipeline from app state
    if not hasattr(request.app.state, 'generator') or request.app.state.generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")

    pipeline = request.app.state.generator.scoring
    if not pipeline or not pipeline.admet_ready:
        raise HTTPException(status_code=503, detail="ADMET predictor not available")

    predictions = pipeline.admet_predictor.predict_all(molecule.smiles)

    return {
        "molecule_id": molecule_id,
        "smiles": molecule.smiles,
        "admet_predictions": predictions
    }


@router.post("/admet/predict")
async def predict_admet(
    admet_request: ADMETPredictRequest,
    request: Request
):
    """Predict ADMET properties for an arbitrary SMILES string."""
    if not hasattr(request.app.state, 'generator') or request.app.state.generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")

    pipeline = request.app.state.generator.scoring
    if not pipeline or not pipeline.admet_ready:
        raise HTTPException(status_code=503, detail="ADMET predictor not available")

    smiles = admet_request.smiles
    properties = admet_request.properties

    if properties:
        predictions = {
            p: pipeline.admet_predictor.predict(smiles, p)
            for p in properties
        }
    else:
        predictions = pipeline.admet_predictor.predict_all(smiles)

    return {
        "smiles": smiles,
        "predictions": predictions
    }


@router.get("/admet/models")
async def list_admet_models(request: Request):
    """List all available ADMET models and their status."""
    if not hasattr(request.app.state, 'generator') or request.app.state.generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")

    pipeline = request.app.state.generator.scoring
    if not pipeline or not pipeline.admet_predictor:
        raise HTTPException(status_code=503, detail="ADMET predictor not available")

    return {
        "models": pipeline.admet_predictor.get_model_info(),
        "loaded_count": len(pipeline.admet_predictor.available_properties),
        "available_properties": pipeline.admet_predictor.available_properties
    }


# MIC Prediction Endpoints

class MICPredictRequest(BaseModel):
    smiles: str
    organism: Optional[str] = "general"


@router.get("/{molecule_id}/mic")
async def get_molecule_mic(
    molecule_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Get detailed MIC predictions for a molecule against all target organisms."""
    molecule = await queries.get_molecule_by_id(db, molecule_id)
    if not molecule:
        raise HTTPException(status_code=404, detail="Molecule not found")

    # Get the scoring pipeline from app state
    if not hasattr(request.app.state, 'generator') or request.app.state.generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")

    pipeline = request.app.state.generator.scoring
    if not pipeline or not pipeline.mic_predictor_ready:
        raise HTTPException(status_code=503, detail="MIC predictor not available")

    predictions = pipeline.efficacy_scorer.mic_predictor.predict_all_organisms(molecule.smiles)

    return {
        "molecule_id": molecule_id,
        "smiles": molecule.smiles,
        "predictions": {
            org: {
                "mic": pred.predicted_mic,
                "confidence": pred.confidence,
                "activity_class": pred.activity_class,
                "percentile": pred.percentile_rank,
                "similar_compounds": pred.similar_compounds
            }
            for org, pred in predictions.items()
        }
    }


@router.post("/mic/predict")
async def predict_mic(
    mic_request: MICPredictRequest,
    request: Request
):
    """Predict MIC for any SMILES string against a specified organism."""
    if not hasattr(request.app.state, 'generator') or request.app.state.generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")

    pipeline = request.app.state.generator.scoring
    if not pipeline or not pipeline.mic_predictor_ready:
        raise HTTPException(status_code=503, detail="MIC predictor not available")

    smiles = mic_request.smiles
    organism = mic_request.organism or "general"

    prediction = pipeline.efficacy_scorer.mic_predictor.predict(smiles, organism)

    return {
        "smiles": smiles,
        "organism": organism,
        "predicted_mic": prediction.predicted_mic,
        "confidence": prediction.confidence,
        "activity_class": prediction.activity_class,
        "percentile_rank": prediction.percentile_rank,
        "similar_compounds": prediction.similar_compounds
    }


@router.post("/mic/predict-all")
async def predict_mic_all_organisms(
    mic_request: MICPredictRequest,
    request: Request
):
    """Predict MIC for a SMILES string against all target organisms."""
    if not hasattr(request.app.state, 'generator') or request.app.state.generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")

    pipeline = request.app.state.generator.scoring
    if not pipeline or not pipeline.mic_predictor_ready:
        raise HTTPException(status_code=503, detail="MIC predictor not available")

    predictions = pipeline.efficacy_scorer.mic_predictor.predict_all_organisms(mic_request.smiles)

    return {
        "smiles": mic_request.smiles,
        "predictions": {
            org: {
                "mic": pred.predicted_mic,
                "confidence": pred.confidence,
                "activity_class": pred.activity_class,
                "percentile": pred.percentile_rank,
                "similar_compounds": pred.similar_compounds
            }
            for org, pred in predictions.items()
        }
    }


@router.get("/mic/status")
async def get_mic_predictor_status(request: Request):
    """Get MIC predictor status and available organisms."""
    if not hasattr(request.app.state, 'generator') or request.app.state.generator is None:
        return {"available": False, "reason": "Generator not initialized"}

    pipeline = request.app.state.generator.scoring
    if not pipeline:
        return {"available": False, "reason": "Scoring pipeline not initialized"}

    if not pipeline.mic_predictor_ready:
        return {"available": False, "reason": "MIC predictor not ready"}

    mic_predictor = pipeline.efficacy_scorer.mic_predictor

    return {
        "available": True,
        "is_trained": mic_predictor.is_trained,
        "organisms": mic_predictor.ORGANISMS,
        "reference_compounds_count": len(mic_predictor.reference_embeddings),
        "has_statistics": len(mic_predictor.mic_statistics) > 0,
        "activity_thresholds": mic_predictor.MIC_THRESHOLDS
    }


# Synthesis Analysis Endpoints

def _get_synthesis_recommendations(sa_result: dict) -> List[str]:
    """Generate synthesis recommendations based on SA analysis"""
    recommendations = []

    components = sa_result.get("components", {})
    problematic = sa_result.get("problematic_groups", [])

    if components.get("stereo_penalty", 0) > 1:
        recommendations.append("Consider racemic synthesis or asymmetric catalysis for stereocenters")

    if components.get("ring_penalty", 0) > 1:
        recommendations.append("Complex ring systems may require multi-step synthesis")

    if components.get("estimated_steps", 0) > 8:
        recommendations.append("Long synthetic route - consider retrosynthetic simplification")

    if "azide" in problematic:
        recommendations.append("Azide groups require careful handling - consider click chemistry")

    if "peroxide" in problematic:
        recommendations.append("Peroxide groups are unstable - consider alternative oxidation states")

    if "epoxide" in problematic:
        recommendations.append("Epoxide groups are reactive - plan ring-opening carefully")

    if "aziridine" in problematic:
        recommendations.append("Aziridine groups are strained - handle with care")

    quat_score = components.get("quat_synthesis", 50)
    if quat_score > 70:
        recommendations.append("Favorable for quaternization via Menshutkin reaction")
    elif quat_score < 40:
        recommendations.append("Quaternization may be challenging - consider simpler alkyl groups")

    if not recommendations:
        recommendations.append("Standard synthetic approaches should be applicable")

    return recommendations


class SynthesisAnalyzeRequest(BaseModel):
    smiles: str


@router.get("/{molecule_id}/synthesis")
async def get_synthesis_analysis(
    molecule_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Get detailed synthetic accessibility analysis for a molecule."""
    molecule = await queries.get_molecule_by_id(db, molecule_id)
    if not molecule:
        raise HTTPException(status_code=404, detail="Molecule not found")

    if not hasattr(request.app.state, 'generator') or request.app.state.generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")

    pipeline = request.app.state.generator.scoring
    if not pipeline or not pipeline.sa_scorer:
        raise HTTPException(status_code=503, detail="SA scorer not available")

    sa_result = await pipeline.sa_scorer.score(molecule.smiles)

    return {
        "molecule_id": molecule_id,
        "smiles": molecule.smiles,
        "sa_score": sa_result.get("score"),
        "sa_score_raw": sa_result.get("components", {}).get("sa_score_raw"),
        "components": sa_result.get("components", {}),
        "problematic_groups": sa_result.get("problematic_groups", []),
        "confidence": sa_result.get("confidence", 0.5),
        "recommendations": _get_synthesis_recommendations(sa_result)
    }


@router.post("/synthesis/analyze")
async def analyze_synthesis(
    synthesis_request: SynthesisAnalyzeRequest,
    request: Request
):
    """Analyze synthetic accessibility for any SMILES string."""
    if not hasattr(request.app.state, 'generator') or request.app.state.generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialized")

    pipeline = request.app.state.generator.scoring
    if not pipeline or not pipeline.sa_scorer:
        raise HTTPException(status_code=503, detail="SA scorer not available")

    sa_result = await pipeline.sa_scorer.score(synthesis_request.smiles)

    return {
        "smiles": synthesis_request.smiles,
        "sa_score": sa_result.get("score"),
        "sa_score_raw": sa_result.get("components", {}).get("sa_score_raw"),
        "components": sa_result.get("components", {}),
        "problematic_groups": sa_result.get("problematic_groups", []),
        "confidence": sa_result.get("confidence", 0.5),
        "recommendations": _get_synthesis_recommendations(sa_result)
    }


@router.get("/synthesis/interpretation")
async def get_sa_score_interpretation():
    """Get interpretation guide for SA scores."""
    return {
        "scale": {
            "raw_sa_score": "1-10 scale where 1=very easy, 10=very hard",
            "normalized_score": "0-100 scale where 100=very easy, 0=very hard"
        },
        "thresholds": {
            "90-100": {"raw": "1-2", "description": "Very easy - simple compounds"},
            "70-90": {"raw": "2-4", "description": "Easy - drug-like complexity"},
            "45-70": {"raw": "4-6", "description": "Moderate - typical pharma complexity"},
            "25-45": {"raw": "6-8", "description": "Difficult - complex synthesis required"},
            "0-25": {"raw": "8-10", "description": "Very difficult - natural product-like complexity"}
        },
        "penalties": {
            "stereo_penalty": "0.5 per chiral center, 0.25 per stereo double bond",
            "macrocycle_penalty": "0.3 * (ring_size - 8) for rings > 8 atoms",
            "spiro_penalty": "0.75 per spiro center",
            "bridged_penalty": "1.0 per bridged ring system"
        },
        "quat_synthesis_interpretation": {
            "70-100": "Favorable for Menshutkin quaternization",
            "40-70": "Standard quaternization should work",
            "0-40": "Quaternization may be challenging"
        }
    }


# ============================================================================
# Molecular Filtering and Diversity Endpoints
# ============================================================================

class FilterBatchRequest(BaseModel):
    smiles_list: List[str]


class FilterSingleRequest(BaseModel):
    smiles: str


class DiversitySelectRequest(BaseModel):
    smiles_list: List[str]
    scores: Optional[List[float]] = None
    n_select: int = Field(50, ge=1, le=1000)
    strategy: str = Field("maxmin", pattern="^(maxmin|leader)$")


class DiversityMetricsRequest(BaseModel):
    smiles_list: List[str]


@router.post("/filter")
async def filter_molecules_batch(
    filter_request: FilterBatchRequest,
    request: Request
):
    """
    Filter a batch of molecules and return detailed reports.

    Applies validity, quat-specific, property range, PAINS, and Brenk filters.
    """
    generator = request.app.state.generator

    if not generator or not generator.molecular_filter:
        raise HTTPException(status_code=503, detail="Molecular filter not available")

    passed, reports = generator.molecular_filter.filter_batch(filter_request.smiles_list)
    rejection_summary = generator.molecular_filter.get_rejection_summary(reports)

    return {
        "total": len(filter_request.smiles_list),
        "passed": len(passed),
        "passed_smiles": passed,
        "rejection_summary": rejection_summary,
        "detailed_reports": [
            {
                "smiles": r.smiles,
                "is_valid": r.is_valid,
                "passed_all": r.passed_all,
                "rejection_reasons": r.rejection_reasons,
                "warnings": r.warnings,
                "properties": r.properties
            }
            for r in reports
        ]
    }


@router.post("/filter/single")
async def filter_single_molecule(
    filter_request: FilterSingleRequest,
    request: Request
):
    """
    Get detailed filter report for a single molecule.

    Returns comprehensive validation results including property calculations.
    """
    generator = request.app.state.generator

    if not generator or not generator.molecular_filter:
        raise HTTPException(status_code=503, detail="Molecular filter not available")

    report = generator.molecular_filter.filter_molecule(filter_request.smiles)

    return {
        "smiles": report.smiles,
        "is_valid": report.is_valid,
        "passed_all": report.passed_all,
        "filter_results": {k: v.value for k, v in report.filter_results.items()},
        "rejection_reasons": report.rejection_reasons,
        "warnings": report.warnings,
        "properties": report.properties
    }


@router.post("/diversity/select")
async def select_diverse_molecules(
    diversity_request: DiversitySelectRequest,
    request: Request
):
    """
    Select a diverse subset of molecules using Tanimoto diversity.

    Strategies:
    - maxmin: Maximize minimum distance to selected set (better coverage)
    - leader: Greedy leader-picker (faster)
    """
    generator = request.app.state.generator

    if not generator or not generator.diversity_selector:
        raise HTTPException(status_code=503, detail="Diversity selector not available")

    if diversity_request.scores and len(diversity_request.scores) != len(diversity_request.smiles_list):
        raise HTTPException(status_code=400, detail="Scores length must match SMILES length")

    selected = generator.diversity_selector.select_diverse(
        diversity_request.smiles_list,
        diversity_request.scores,
        diversity_request.n_select,
        diversity_request.strategy
    )

    # Calculate diversity metrics for selected set
    selected_smiles = [s for s, _ in selected]
    metrics = generator.diversity_selector.calculate_diversity_metrics(selected_smiles)

    return {
        "selected_count": len(selected),
        "requested": diversity_request.n_select,
        "input_count": len(diversity_request.smiles_list),
        "selected": [{"smiles": s, "score": sc} for s, sc in selected],
        "diversity_metrics": metrics
    }


@router.post("/diversity/metrics")
async def calculate_diversity_metrics(
    metrics_request: DiversityMetricsRequest,
    request: Request
):
    """
    Calculate diversity metrics for a set of molecules.

    Returns mean, min, max, and std of pairwise Tanimoto distances.
    """
    generator = request.app.state.generator

    if not generator or not generator.diversity_selector:
        raise HTTPException(status_code=503, detail="Diversity selector not available")

    metrics = generator.diversity_selector.calculate_diversity_metrics(
        metrics_request.smiles_list
    )

    return metrics


@router.get("/filter/patterns")
async def get_filter_patterns():
    """
    Get the SMARTS patterns used for filtering.

    Returns quat detection patterns, counterion patterns, and exclusion patterns.
    """
    from generator.filters import MolecularFilter

    return {
        "quat_patterns": MolecularFilter.QUAT_PATTERNS,
        "counterion_patterns": MolecularFilter.COUNTERION_PATTERNS,
        "exclusion_patterns": MolecularFilter.QUAT_EXCLUSIONS
    }
