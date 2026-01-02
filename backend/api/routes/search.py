"""Search API routes for substructure and similarity search"""

from fastapi import APIRouter, Request, Query, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from search.substructure import SubstructureSearch, SearchQuery, CommonPatterns

router = APIRouter()

# Create search engine instance
search_engine = SubstructureSearch()


class SubstructureSearchRequest(BaseModel):
    """Request body for substructure search"""
    smarts: str = Field(..., description="SMARTS pattern to search for")
    max_results: int = Field(default=100, ge=1, le=1000)
    require_quat: bool = Field(default=False, description="Also require quaternary nitrogen")
    min_efficacy: Optional[float] = Field(default=None, ge=0, le=100)
    min_safety: Optional[float] = Field(default=None, ge=0, le=100)
    min_sa: Optional[float] = Field(default=None, ge=0, le=100)


class SimilaritySearchRequest(BaseModel):
    """Request body for similarity search"""
    smiles: str = Field(..., description="Query SMILES for similarity search")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_results: int = Field(default=100, ge=1, le=1000)


class SingleMoleculeSearchRequest(BaseModel):
    """Request for searching a single molecule"""
    smiles: str
    smarts: str


class FilterRequest(BaseModel):
    """Request for filtering molecules by substructure"""
    smiles_list: List[str]
    required_patterns: Optional[List[str]] = None
    excluded_patterns: Optional[List[str]] = None


class SearchResultResponse(BaseModel):
    """Response for search results"""
    smiles: str
    molecule_id: Optional[int] = None
    name: Optional[str] = None
    match_atoms: List[List[int]] = []
    match_count: int = 0
    scores: Dict[str, float] = {}
    metadata: Dict[str, Any] = {}


class SimilarityResultResponse(BaseModel):
    """Response for similarity search results"""
    smiles: str
    molecule_id: Optional[int] = None
    name: Optional[str] = None
    similarity: float
    scores: Dict[str, float] = {}


@router.get("/status")
async def get_search_status():
    """Get search engine status"""
    return {
        "available": search_engine.is_ready,
        "features": {
            "substructure_search": search_engine.is_ready,
            "similarity_search": search_engine.is_ready,
            "pattern_validation": search_engine.is_ready,
            "quat_classification": search_engine.is_ready,
        },
        "common_patterns": len(CommonPatterns.get_all_patterns()),
        "quat_patterns": len(CommonPatterns.get_quat_patterns())
    }


@router.get("/patterns")
async def get_common_patterns():
    """Get all common SMARTS patterns"""
    return {
        "all_patterns": CommonPatterns.get_all_patterns(),
        "quat_patterns": CommonPatterns.get_quat_patterns()
    }


@router.post("/validate")
async def validate_smarts(smarts: str = Query(..., description="SMARTS pattern to validate")):
    """Validate a SMARTS pattern"""
    is_valid, error = search_engine.validate_smarts(smarts)
    return {
        "smarts": smarts,
        "is_valid": is_valid,
        "error": error
    }


@router.post("/substructure", response_model=List[SearchResultResponse])
async def search_substructure(request: SubstructureSearchRequest, req: Request):
    """
    Search database molecules for a substructure pattern.

    Uses SMARTS pattern matching to find molecules containing the specified
    structural motif.
    """
    if not search_engine.is_ready:
        raise HTTPException(status_code=503, detail="Search engine not available")

    # Validate SMARTS
    is_valid, error = search_engine.validate_smarts(request.smarts)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid SMARTS pattern: {error}")

    # Get molecules from database
    from database.connection import get_db_session
    from database.queries import get_all_molecules

    async with get_db_session() as db:
        molecules = await get_all_molecules(db, limit=10000)

    # Convert to dicts for search
    mol_dicts = []
    for mol in molecules:
        mol_dicts.append({
            "id": mol.id,
            "smiles": mol.smiles,
            "name": mol.name,
            "efficacy_score": mol.efficacy_score,
            "safety_score": mol.safety_score,
            "sa_score": mol.sa_score,
            "environmental_score": mol.environmental_score,
            "is_pareto": mol.is_pareto,
            "is_starred": mol.is_starred
        })

    # Build search query
    query = SearchQuery(
        pattern=request.smarts,
        max_results=request.max_results,
        require_quat=request.require_quat,
        min_efficacy=request.min_efficacy,
        min_safety=request.min_safety,
        min_sa=request.min_sa,
        include_metadata=True
    )

    # Execute search
    results = search_engine.search_molecules(mol_dicts, query)

    # Convert to response format
    return [
        SearchResultResponse(
            smiles=r.smiles,
            molecule_id=r.molecule_id,
            name=r.name,
            match_atoms=r.match_atoms,
            match_count=r.match_count,
            scores=r.scores,
            metadata=r.metadata
        )
        for r in results
    ]


@router.post("/similarity", response_model=List[SimilarityResultResponse])
async def search_similarity(request: SimilaritySearchRequest, req: Request):
    """
    Find molecules similar to a query molecule.

    Uses Morgan fingerprints and Tanimoto similarity.
    """
    if not search_engine.is_ready:
        raise HTTPException(status_code=503, detail="Search engine not available")

    # Get molecules from database
    from database.connection import get_db_session
    from database.queries import get_all_molecules

    async with get_db_session() as db:
        molecules = await get_all_molecules(db, limit=10000)

    # Convert to dicts
    mol_dicts = []
    for mol in molecules:
        mol_dicts.append({
            "id": mol.id,
            "smiles": mol.smiles,
            "name": mol.name,
            "efficacy_score": mol.efficacy_score,
            "safety_score": mol.safety_score,
            "sa_score": mol.sa_score
        })

    # Execute similarity search
    results = search_engine.similarity_search(
        request.smiles,
        mol_dicts,
        threshold=request.threshold,
        max_results=request.max_results
    )

    # Convert to response format
    return [
        SimilarityResultResponse(
            smiles=mol_data["smiles"],
            molecule_id=mol_data.get("id"),
            name=mol_data.get("name"),
            similarity=similarity,
            scores={
                k: mol_data[k] for k in ["efficacy_score", "safety_score", "sa_score"]
                if k in mol_data and mol_data[k] is not None
            }
        )
        for mol_data, similarity in results
    ]


@router.post("/check")
async def check_single_molecule(request: SingleMoleculeSearchRequest):
    """
    Check if a single molecule contains a substructure pattern.

    Quick endpoint for validating individual molecules.
    """
    if not search_engine.is_ready:
        raise HTTPException(status_code=503, detail="Search engine not available")

    # Validate SMARTS
    is_valid, error = search_engine.validate_smarts(request.smarts)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid SMARTS pattern: {error}")

    result = search_engine.search_molecule(request.smiles, request.smarts)

    if result:
        return {
            "matches": True,
            "match_count": result.match_count,
            "match_atoms": result.match_atoms
        }
    else:
        return {
            "matches": False,
            "match_count": 0,
            "match_atoms": []
        }


@router.post("/filter")
async def filter_molecules(request: FilterRequest):
    """
    Filter molecules by required and excluded substructures.

    Returns only molecules that match all required patterns
    and don't match any excluded patterns.
    """
    if not search_engine.is_ready:
        raise HTTPException(status_code=503, detail="Search engine not available")

    # Validate patterns
    for pattern in (request.required_patterns or []):
        is_valid, error = search_engine.validate_smarts(pattern)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid required pattern: {error}")

    for pattern in (request.excluded_patterns or []):
        is_valid, error = search_engine.validate_smarts(pattern)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid excluded pattern: {error}")

    filtered = search_engine.filter_by_substructure(
        request.smiles_list,
        required_patterns=request.required_patterns,
        excluded_patterns=request.excluded_patterns
    )

    return {
        "input_count": len(request.smiles_list),
        "output_count": len(filtered),
        "filtered_smiles": filtered
    }


@router.post("/classify")
async def classify_quat_type(smiles: str = Query(..., description="SMILES to classify")):
    """
    Classify the type of quaternary ammonium compound.

    Returns the classification (benzalkonium, didecyl, cetyl, pyridinium,
    imidazolium, other_quat) or null if not a quat.
    """
    if not search_engine.is_ready:
        raise HTTPException(status_code=503, detail="Search engine not available")

    quat_type = search_engine.classify_quat_type(smiles)

    return {
        "smiles": smiles,
        "is_quat": quat_type is not None,
        "quat_type": quat_type
    }


@router.post("/features")
async def get_structural_features(smiles: str = Query(..., description="SMILES to analyze")):
    """
    Get structural feature summary for a molecule.

    Returns presence of common functional groups and structural motifs.
    """
    if not search_engine.is_ready:
        raise HTTPException(status_code=503, detail="Search engine not available")

    features = search_engine.get_structural_features(smiles)

    return {
        "smiles": smiles,
        "features": features
    }


@router.post("/scaffolds")
async def find_scaffolds(smiles_list: List[str]):
    """
    Find which molecules match common quaternary ammonium scaffolds.

    Groups molecules by their scaffold type.
    """
    if not search_engine.is_ready:
        raise HTTPException(status_code=503, detail="Search engine not available")

    scaffolds = search_engine.find_common_scaffolds(smiles_list)

    # Add counts
    scaffold_summary = {
        name: {
            "count": len(matches),
            "smiles": matches[:10]  # Limit to first 10 examples
        }
        for name, matches in scaffolds.items()
    }

    return {
        "total_molecules": len(smiles_list),
        "scaffolds": scaffold_summary
    }


@router.get("/quat-check")
async def check_has_quat(smiles: str = Query(..., description="SMILES to check")):
    """Quick check if molecule has quaternary nitrogen"""
    if not search_engine.is_ready:
        raise HTTPException(status_code=503, detail="Search engine not available")

    has_quat = search_engine.has_quat_nitrogen(smiles)

    return {
        "smiles": smiles,
        "has_quat_nitrogen": has_quat
    }
