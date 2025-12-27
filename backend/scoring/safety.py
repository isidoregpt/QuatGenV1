"""
Safety Scoring
Predict human toxicity of quaternary ammonium compounds

Components:
- Acute oral toxicity (LD50)
- Skin irritation
- Eye irritation
- Respiratory sensitization
"""

import logging
from typing import Dict
import math

logger = logging.getLogger(__name__)


class SafetyScorer:
    """
    Human safety scorer
    
    Uses QSAR models to predict toxicity endpoints.
    Higher score = safer compound.
    """
    
    def __init__(self):
        self._is_ready = False
    
    async def initialize(self):
        """Initialize scoring model"""
        self._is_ready = True
        logger.info("Safety scorer initialized (QSAR mode)")
    
    async def score(self, smiles: str) -> Dict:
        """
        Score human safety (higher = safer)
        
        Returns dict with:
        - score: 0-100 overall safety
        - components: Individual hazard scores
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"score": 0, "components": {}, "error": "Invalid SMILES"}
            
            # Calculate properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            components = {}
            
            # 1. Acute oral toxicity
            # Quats generally have moderate oral toxicity
            # LD50 typically 200-2000 mg/kg
            # Lower logP = better oral safety
            oral_safety = self._score_oral_toxicity(logp, mw)
            components["acute_toxicity"] = oral_safety
            
            # 2. Skin irritation
            # Higher logP = more irritating
            # Shorter chains = less irritating
            skin_safety = self._score_skin_irritation(logp, mw)
            components["skin_irritation"] = skin_safety
            
            # 3. Eye irritation
            # Quats are generally eye irritants
            # Can estimate based on charge and logP
            eye_safety = self._score_eye_irritation(mol, logp)
            components["eye_irritation"] = eye_safety
            
            # 4. Respiratory sensitization
            # Based on volatility (MW) and reactivity
            resp_safety = self._score_respiratory(mw, logp)
            components["respiratory"] = resp_safety
            
            # Overall safety (weighted average)
            weights = {
                "acute_toxicity": 0.3,
                "skin_irritation": 0.3,
                "eye_irritation": 0.25,
                "respiratory": 0.15
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
            logger.error(f"Safety scoring error: {e}")
            return {"score": 0, "components": {}, "error": str(e)}
    
    def _score_oral_toxicity(self, logp: float, mw: float) -> float:
        """
        Predict oral toxicity safety score
        
        Based on EPA ECOSAR-like predictions.
        Lower logP and higher MW generally = lower acute toxicity
        """
        # LogP contribution (lower = safer)
        if logp < 2:
            logp_score = 90
        elif logp < 4:
            logp_score = 80 - (logp - 2) * 10
        elif logp < 6:
            logp_score = 60 - (logp - 4) * 15
        else:
            logp_score = 30
        
        # MW contribution (moderate MW = safer)
        if 300 <= mw <= 500:
            mw_score = 80
        elif 200 <= mw < 300:
            mw_score = 70
        elif 500 < mw <= 700:
            mw_score = 60
        else:
            mw_score = 50
        
        return (logp_score + mw_score) / 2
    
    def _score_skin_irritation(self, logp: float, mw: float) -> float:
        """
        Predict skin irritation safety score
        
        Quats can cause skin irritation, especially at high concentrations.
        Lower lipophilicity = less penetration = less irritation
        """
        # LogP contribution
        if logp < 1:
            logp_score = 95
        elif logp < 3:
            logp_score = 85 - (logp - 1) * 10
        elif logp < 5:
            logp_score = 65 - (logp - 3) * 15
        else:
            logp_score = 35
        
        # MW contribution (higher MW = less penetration)
        mw_factor = min(1.2, mw / 400)
        
        return min(100, logp_score * mw_factor)
    
    def _score_eye_irritation(self, mol, logp: float) -> float:
        """
        Predict eye irritation safety score
        
        Cationic surfactants are typically eye irritants.
        Score based on charge density and lipophilicity.
        """
        # Get formal charge
        charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        
        # Higher charge = more irritating
        if charge == 0:
            charge_score = 90
        elif charge == 1:
            charge_score = 60
        else:
            charge_score = 40
        
        # LogP contribution
        if logp < 2:
            logp_score = 70
        elif logp < 4:
            logp_score = 55
        else:
            logp_score = 40
        
        return (charge_score + logp_score) / 2
    
    def _score_respiratory(self, mw: float, logp: float) -> float:
        """
        Predict respiratory safety score
        
        Higher MW = less volatile = safer for inhalation
        """
        # MW contribution (higher = safer)
        if mw > 400:
            mw_score = 90
        elif mw > 300:
            mw_score = 80
        elif mw > 200:
            mw_score = 65
        else:
            mw_score = 50
        
        # LogP contribution (lower = safer for respiratory)
        if logp < 2:
            logp_score = 85
        elif logp < 4:
            logp_score = 70
        else:
            logp_score = 55
        
        return (mw_score + logp_score) / 2
