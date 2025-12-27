"""
Efficacy Scoring
Predict antimicrobial activity of quaternary ammonium compounds

Components:
- Gram-positive MIC prediction
- Gram-negative MIC prediction
- Antifungal activity
- CMC (Critical Micelle Concentration)
- Membrane disruption potential
"""

import logging
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EfficacyScorer:
    """
    Antimicrobial efficacy scorer
    
    Uses QSAR models to predict antimicrobial activity.
    For quats, key factors are:
    - Alkyl chain length (optimal C12-C14)
    - logP (lipophilicity)
    - Charge distribution
    """
    
    def __init__(self):
        self._is_ready = False
        self.mol = None  # RDKit molecule
    
    async def initialize(self):
        """Initialize scoring model"""
        # For MVP: use QSAR rules
        # For production: load trained ML model
        self._is_ready = True
        logger.info("Efficacy scorer initialized (QSAR mode)")
    
    async def score(self, smiles: str) -> Dict:
        """
        Score antimicrobial efficacy
        
        Returns dict with:
        - score: 0-100 overall efficacy
        - components: Individual component scores
        - mw, logp, chain_length: Molecular properties
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
            tpsa = Descriptors.TPSA(mol)
            
            # Estimate chain length (count consecutive aliphatic carbons)
            chain_length = self._estimate_chain_length(mol)
            
            # Component scores
            components = {}
            
            # 1. Gram-positive MIC prediction
            # Optimal logP around 4-6 for quats
            gram_pos_score = self._score_logp_range(logp, optimal_min=3, optimal_max=6)
            # Chain length bonus (C12-C14 optimal)
            gram_pos_score *= self._score_chain_length(chain_length, optimal=12)
            components["gram_positive_mic"] = min(100, gram_pos_score * 100)
            
            # 2. Gram-negative MIC prediction
            # Need higher lipophilicity for gram-negative
            gram_neg_score = self._score_logp_range(logp, optimal_min=4, optimal_max=7)
            gram_neg_score *= self._score_chain_length(chain_length, optimal=14)
            components["gram_negative_mic"] = min(100, gram_neg_score * 100)
            
            # 3. Antifungal activity
            # Similar to gram-positive but with different optimum
            antifungal_score = self._score_logp_range(logp, optimal_min=4, optimal_max=8)
            antifungal_score *= self._score_chain_length(chain_length, optimal=16)
            components["antifungal"] = min(100, antifungal_score * 100)
            
            # 4. CMC score (lower is better for activity)
            # CMC inversely related to chain length
            cmc_score = min(1.0, chain_length / 16) * 100
            components["cmc_score"] = cmc_score
            
            # 5. Membrane disruption potential
            # Based on amphiphilicity (logP + positive charge)
            membrane_score = self._score_membrane_disruption(mol, logp)
            components["membrane_disruption"] = membrane_score
            
            # Calculate overall score (weighted average)
            weights = {
                "gram_positive_mic": 0.25,
                "gram_negative_mic": 0.25,
                "antifungal": 0.2,
                "cmc_score": 0.15,
                "membrane_disruption": 0.15
            }
            
            overall = sum(
                components[k] * weights[k]
                for k in weights
            )
            
            return {
                "score": round(overall, 1),
                "components": components,
                "mw": round(mw, 1),
                "logp": round(logp, 2),
                "chain_length": chain_length
            }
            
        except Exception as e:
            logger.error(f"Efficacy scoring error: {e}")
            return {"score": 0, "components": {}, "error": str(e)}
    
    def _estimate_chain_length(self, mol) -> int:
        """Estimate longest alkyl chain length"""
        try:
            # Count aliphatic carbons not in rings
            aliphatic_c = sum(
                1 for atom in mol.GetAtoms()
                if atom.GetSymbol() == "C" 
                and not atom.GetIsAromatic()
                and not atom.IsInRing()
            )
            return aliphatic_c
        except Exception:
            return 0
    
    def _score_logp_range(
        self,
        logp: float,
        optimal_min: float,
        optimal_max: float
    ) -> float:
        """Score based on logP being in optimal range"""
        if optimal_min <= logp <= optimal_max:
            return 1.0
        elif logp < optimal_min:
            return max(0, 1 - (optimal_min - logp) / 3)
        else:
            return max(0, 1 - (logp - optimal_max) / 3)
    
    def _score_chain_length(self, length: int, optimal: int) -> float:
        """Score based on chain length proximity to optimal"""
        diff = abs(length - optimal)
        if diff == 0:
            return 1.0
        elif diff <= 2:
            return 0.9
        elif diff <= 4:
            return 0.7
        else:
            return max(0.3, 1 - diff * 0.1)
    
    def _score_membrane_disruption(self, mol, logp: float) -> float:
        """Score membrane disruption potential"""
        # Check for positive charge
        charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        
        if charge < 1:
            return 30  # Need positive charge
        
        # Amphiphilicity score
        if 3 <= logp <= 8:
            amphiphilicity = 80 + min(20, (logp - 3) * 4)
        else:
            amphiphilicity = 50
        
        return amphiphilicity
