"""Efficacy Scoring - Predict antimicrobial activity"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class EfficacyScorer:
    def __init__(self):
        self._is_ready = False
    
    async def initialize(self):
        self._is_ready = True
        logger.info("Efficacy scorer initialized (QSAR mode)")
    
    async def score(self, smiles: str) -> Dict:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"score": 0, "components": {}, "error": "Invalid SMILES"}
            
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            chain_length = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C" and not a.GetIsAromatic() and not a.IsInRing())
            
            components = {}
            # Gram-positive: optimal logP 3-6, chain C12
            gram_pos = self._score_range(logp, 3, 6) * self._score_chain(chain_length, 12) * 100
            components["gram_positive_mic"] = min(100, gram_pos)
            # Gram-negative: optimal logP 4-7, chain C14
            gram_neg = self._score_range(logp, 4, 7) * self._score_chain(chain_length, 14) * 100
            components["gram_negative_mic"] = min(100, gram_neg)
            # Antifungal: optimal logP 4-8, chain C16
            antifungal = self._score_range(logp, 4, 8) * self._score_chain(chain_length, 16) * 100
            components["antifungal"] = min(100, antifungal)
            # CMC
            components["cmc_score"] = min(100, chain_length / 16 * 100)
            # Membrane disruption
            charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
            components["membrane_disruption"] = 80 if charge >= 1 and 3 <= logp <= 8 else 50
            
            overall = sum(v * w for v, w in zip(
                [components["gram_positive_mic"], components["gram_negative_mic"], components["antifungal"],
                 components["cmc_score"], components["membrane_disruption"]],
                [0.25, 0.25, 0.2, 0.15, 0.15]
            ))
            return {"score": round(overall, 1), "components": components, "mw": round(mw, 1), "logp": round(logp, 2), "chain_length": chain_length}
        except Exception as e:
            logger.error(f"Efficacy scoring error: {e}")
            return {"score": 0, "components": {}, "error": str(e)}
    
    def _score_range(self, val: float, min_v: float, max_v: float) -> float:
        if min_v <= val <= max_v:
            return 1.0
        return max(0, 1 - abs(val - (min_v + max_v) / 2) / 3)
    
    def _score_chain(self, length: int, optimal: int) -> float:
        diff = abs(length - optimal)
        return max(0.3, 1 - diff * 0.1)
