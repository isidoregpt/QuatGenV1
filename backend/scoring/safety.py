"""Safety Scoring - Predict human toxicity"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class SafetyScorer:
    def __init__(self):
        self._is_ready = False
    
    async def initialize(self):
        self._is_ready = True
        logger.info("Safety scorer initialized (QSAR mode)")
    
    async def score(self, smiles: str) -> Dict:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"score": 0, "components": {}, "error": "Invalid SMILES"}
            
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            
            components = {}
            
            # Oral toxicity - lower logP safer
            if logp < 2:
                oral = 90
            elif logp < 4:
                oral = 70
            else:
                oral = max(30, 60 - (logp - 4) * 10)
            components["acute_toxicity"] = oral
            
            # Skin irritation - lower logP less penetration
            if logp < 1:
                skin = 95
            elif logp < 3:
                skin = 75
            else:
                skin = max(35, 65 - (logp - 3) * 10)
            components["skin_irritation"] = skin
            
            # Eye irritation - quats are generally irritants
            charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
            if charge == 0:
                eye = 90
            elif charge == 1:
                eye = 60
            else:
                eye = 40
            components["eye_irritation"] = eye
            
            # Respiratory - higher MW safer
            if mw > 400:
                resp = 90
            elif mw > 300:
                resp = 75
            else:
                resp = 60
            components["respiratory"] = resp
            
            overall = (components["acute_toxicity"] * 0.3 + components["skin_irritation"] * 0.3 +
                      components["eye_irritation"] * 0.25 + components["respiratory"] * 0.15)
            
            return {"score": round(overall, 1), "components": components}
        except Exception as e:
            logger.error(f"Safety scoring error: {e}")
            return {"score": 0, "components": {}, "error": str(e)}
