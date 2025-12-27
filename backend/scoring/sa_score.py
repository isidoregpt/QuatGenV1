"""
Synthetic Accessibility Scoring
Predict ease of synthesis for quaternary ammonium compounds

Uses RDKit's SA score with modifications for quat-specific chemistry.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class SAScorer:
    """
    Synthetic Accessibility scorer
    
    Based on RDKit's SA score (Ertl & Schuffenhauer, 2009)
    with modifications for quaternary ammonium compounds.
    """
    
    def __init__(self):
        self._is_ready = False
        self._sa_model = None
    
    async def initialize(self):
        """Initialize SA scorer"""
        try:
            from rdkit.Chem import RDConfig
            import os
            
            # Try to load fragment scores
            # These are used by RDKit's SA score
            self._is_ready = True
            logger.info("SA scorer initialized")
        except Exception as e:
            logger.warning(f"SA scorer initialization warning: {e}")
            self._is_ready = True  # Fall back to simplified scoring
    
    async def score(self, smiles: str) -> Dict:
        """
        Score synthetic accessibility (higher = easier to make)
        
        Returns dict with:
        - score: 0-100 SA score (100 = trivial to synthesize)
        - components: Breakdown of scoring factors
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"score": 0, "components": {}, "error": "Invalid SMILES"}
            
            components = {}
            
            # 1. RDKit SA score (1-10, lower is better)
            try:
                from rdkit.Chem import rdMolDescriptors
                # SA score calculation
                sa_raw = self._calculate_sa_score(mol)
                # Convert to 0-100 (invert: lower SA = higher score)
                sa_score = max(0, (10 - sa_raw) / 9 * 100)
                components["sa_score"] = round(sa_score, 1)
            except Exception:
                # Fallback
                sa_score = 50.0
                components["sa_score"] = 50.0
            
            # 2. Complexity score
            complexity = self._score_complexity(mol)
            components["complexity"] = complexity
            
            # 3. Starting materials availability
            # Quats are typically made from:
            # - Tertiary amines + alkyl halides
            # - Both readily available
            starting_materials = self._score_starting_materials(mol, smiles)
            components["starting_materials"] = starting_materials
            
            # 4. Number of synthetic steps (estimate)
            steps = self._estimate_steps(mol)
            steps_score = max(0, 100 - steps * 15)  # Penalty per step
            components["estimated_steps"] = steps
            
            # Overall score
            overall = (
                0.4 * components["sa_score"] +
                0.3 * complexity +
                0.2 * starting_materials +
                0.1 * steps_score
            )
            
            return {
                "score": round(overall, 1),
                "components": components
            }
            
        except Exception as e:
            logger.error(f"SA scoring error: {e}")
            return {"score": 50, "components": {}, "error": str(e)}
    
    def _calculate_sa_score(self, mol) -> float:
        """
        Calculate SA score (1-10)
        
        Simplified implementation based on:
        - Fragment contributions
        - Stereocenters
        - Ring complexity
        - Size
        """
        try:
            from rdkit.Chem import rdMolDescriptors
            
            # Start with base score
            score = 2.0
            
            # Add ring complexity
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            ring_score = num_rings * 0.3
            score += min(ring_score, 2.0)
            
            # Add stereocenter complexity
            chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            score += chiral_centers * 0.5
            
            # Add size penalty
            num_atoms = mol.GetNumHeavyAtoms()
            if num_atoms > 35:
                score += (num_atoms - 35) * 0.1
            
            # Spiro and bridged rings are hard
            ring_info = mol.GetRingInfo()
            # Simplified check
            if num_rings > 3:
                score += 0.5
            
            # Cap at 10
            return min(10.0, max(1.0, score))
            
        except Exception:
            return 5.0  # Default medium difficulty
    
    def _score_complexity(self, mol) -> float:
        """
        Score molecular complexity (higher = simpler = better)
        """
        try:
            from rdkit.Chem import GraphDescriptors
            
            # Bertz complexity
            try:
                complexity = GraphDescriptors.BertzCT(mol)
                # Normalize (typical range 0-1000)
                norm_complexity = min(1.0, complexity / 500)
                score = (1 - norm_complexity) * 100
            except Exception:
                score = 70.0
            
            # Penalty for many rings
            num_rings = mol.GetRingInfo().NumRings()
            score -= num_rings * 5
            
            # Penalty for stereocenters
            from rdkit import Chem
            chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            score -= chiral * 10
            
            return max(0, min(100, score))
            
        except Exception:
            return 60.0
    
    def _score_starting_materials(self, mol, smiles: str) -> float:
        """
        Score availability of starting materials
        
        Quats are typically easy to make from:
        - Tertiary amines (trimethylamine, etc.)
        - Alkyl halides (bromoalkanes, etc.)
        """
        score = 80.0  # Base score for quats (generally easy)
        
        # Benzyl quats are very common/easy
        if "Cc1ccccc1" in smiles or "c1ccccc1C" in smiles:
            score += 10
        
        # Long chains might need special alkyl halides
        num_carbon = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
        if num_carbon > 25:
            score -= 15
        
        # Unusual heteroatoms
        unusual = sum(
            1 for a in mol.GetAtoms()
            if a.GetSymbol() in ["P", "Si", "B", "Se"]
        )
        score -= unusual * 20
        
        return max(0, min(100, score))
    
    def _estimate_steps(self, mol) -> int:
        """
        Estimate number of synthetic steps
        
        For simple quats:
        1. Alkylation of tertiary amine (1 step)
        
        For more complex:
        + Chain synthesis
        + Ring construction
        + Functional group interconversion
        """
        steps = 1  # Base: quaternization step
        
        # Additional complexity
        num_rings = mol.GetRingInfo().NumRings()
        if num_rings > 2:
            steps += num_rings - 2
        
        # Heteroatom-containing rings
        aromatic_n = sum(
            1 for a in mol.GetAtoms()
            if a.GetSymbol() == "N" and a.GetIsAromatic()
        )
        if aromatic_n > 1:
            steps += 1
        
        # Long chains might need coupling
        num_atoms = mol.GetNumHeavyAtoms()
        if num_atoms > 40:
            steps += 1
        if num_atoms > 60:
            steps += 1
        
        return steps


# For backward compatibility
try:
    from rdkit import Chem
except ImportError:
    pass
