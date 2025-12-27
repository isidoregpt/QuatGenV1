"""
Environmental Scoring
Predict environmental impact of quaternary ammonium compounds

Components:
- Biodegradability (BIOWIN-like)
- Aquatic toxicity (ECOSAR-like)
- Bioaccumulation (BCF)
- Persistence
"""

import logging
from typing import Dict
import math

logger = logging.getLogger(__name__)


class EnvironmentalScorer:
    """
    Environmental impact scorer
    
    Based on EPA EPI Suite methodology:
    - BIOWIN for biodegradability
    - ECOSAR for aquatic toxicity
    - BCFBAF for bioaccumulation
    
    Higher score = better environmental profile.
    """
    
    def __init__(self):
        self._is_ready = False
    
    async def initialize(self):
        """Initialize scoring model"""
        self._is_ready = True
        logger.info("Environmental scorer initialized (QSAR mode)")
    
    async def score(self, smiles: str) -> Dict:
        """
        Score environmental impact (higher = better)
        
        Returns dict with:
        - score: 0-100 overall environmental score
        - components: Individual component scores
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"score": 0, "components": {}, "error": "Invalid SMILES"}
            
            # Calculate properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            
            # Count functional groups
            num_n = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
            num_o = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            num_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
            
            components = {}
            
            # 1. Biodegradability (BIOWIN-like)
            biodeg = self._score_biodegradability(mol, logp, mw, num_n, num_o)
            components["biodegradability"] = biodeg
            
            # 2. Aquatic toxicity (ECOSAR-like)
            # Lower toxicity = higher score
            aquatic = self._score_aquatic_toxicity(logp, mw)
            components["aquatic_toxicity"] = aquatic
            
            # 3. Bioaccumulation (BCF)
            # Lower BCF = higher score
            bcf = self._score_bioaccumulation(logp)
            components["bioaccumulation"] = bcf
            
            # 4. Persistence
            persistence = self._score_persistence(mol, logp, num_aromatic)
            components["persistence"] = persistence
            
            # Overall environmental score
            weights = {
                "biodegradability": 0.35,
                "aquatic_toxicity": 0.30,
                "bioaccumulation": 0.20,
                "persistence": 0.15
            }
            
            overall = sum(
                components[k] * weights[k]
                for k in weights
            )
            
            return {
                "score": round(overall, 1),
                "components": components
            }
            
        except Exception as e:
            logger.error(f"Environmental scoring error: {e}")
            return {"score": 0, "components": {}, "error": str(e)}
    
    def _score_biodegradability(
        self,
        mol,
        logp: float,
        mw: float,
        num_n: int,
        num_o: int
    ) -> float:
        """
        Predict biodegradability (BIOWIN-like)
        
        Factors favoring biodegradation:
        - Linear alkyl chains (most quats have these)
        - Ester/amide bonds (cleavable)
        - Moderate MW
        
        Factors hindering:
        - Aromatic rings
        - High branching
        - Quaternary carbon centers
        """
        score = 50.0  # Base score
        
        # LogP contribution
        # Moderate logP (2-5) is optimal for biodegradation
        if 2 <= logp <= 5:
            score += 15
        elif logp < 2:
            score += 10
        elif logp <= 7:
            score += 5
        else:
            score -= 10
        
        # MW contribution
        # Lower MW = easier to biodegrade
        if mw < 300:
            score += 15
        elif mw < 400:
            score += 10
        elif mw < 500:
            score += 5
        else:
            score -= 5
        
        # Heteroatom contribution
        # O and N can help biodegradation (cleavage sites)
        if num_o >= 2:
            score += 10
        if num_n >= 2:
            score += 5
        
        # Aromatic ring penalty
        num_aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        score -= num_aromatic * 3
        
        # Quaternary nitrogen is generally biodegradable
        # (unlike quaternary carbon)
        score += 5
        
        return max(0, min(100, score))
    
    def _score_aquatic_toxicity(self, logp: float, mw: float) -> float:
        """
        Predict aquatic toxicity safety (ECOSAR-like)
        
        Cationic surfactants are toxic to aquatic organisms.
        Score represents SAFETY (higher = less toxic = better).
        
        Based on fish LC50 predictions.
        """
        # LogP-based toxicity
        # Higher logP = higher toxicity = lower safety score
        if logp < 2:
            logp_score = 80
        elif logp < 4:
            logp_score = 60 - (logp - 2) * 10
        elif logp < 6:
            logp_score = 40 - (logp - 4) * 10
        else:
            logp_score = 20
        
        # MW modulation
        # Higher MW = lower bioavailability = less toxic
        mw_factor = min(1.3, mw / 350)
        
        return min(100, logp_score * mw_factor)
    
    def _score_bioaccumulation(self, logp: float) -> float:
        """
        Predict bioaccumulation potential (BCF-based)
        
        Score represents LOW bioaccumulation (higher = better).
        
        BCF is strongly correlated with logP.
        """
        # Estimate log BCF from logP
        # BCF = 0.85 * logP - 0.70 (simplified)
        log_bcf = 0.85 * logp - 0.70
        
        # Convert to safety score
        # Low BCF (log BCF < 2) = good
        # High BCF (log BCF > 3.5) = bad
        if log_bcf < 1:
            score = 95
        elif log_bcf < 2:
            score = 85
        elif log_bcf < 3:
            score = 65
        elif log_bcf < 4:
            score = 40
        else:
            score = 20
        
        return score
    
    def _score_persistence(
        self,
        mol,
        logp: float,
        num_aromatic: int
    ) -> float:
        """
        Predict environmental persistence
        
        Score represents LOW persistence (higher = better).
        """
        score = 70.0  # Base score for quats
        
        # Aromatic rings increase persistence
        score -= num_aromatic * 10
        
        # High logP increases persistence
        if logp > 5:
            score -= (logp - 5) * 8
        
        # Halogen atoms increase persistence
        halogens = sum(
            1 for a in mol.GetAtoms()
            if a.GetSymbol() in ["F", "Cl", "Br", "I"]
        )
        score -= halogens * 5
        
        # Ester/amide bonds decrease persistence
        # (hydrolyzable)
        smiles = mol.GetPropsAsDict().get("_smilesAtomOutputOrder", "")
        if "C(=O)O" in str(mol) or "C(=O)N" in str(mol):
            score += 10
        
        return max(0, min(100, score))
