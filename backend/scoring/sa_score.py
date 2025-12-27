"""Synthetic Accessibility Scoring"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, GraphDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None


class SAScorer:
    def __init__(self):
        self._is_ready = False
    
    async def initialize(self):
        self._is_ready = True
        logger.info("SA scorer initialized")
    
    async def score(self, smiles: str) -> Dict:
        if not RDKIT_AVAILABLE:
            return {"score": 50, "components": {}, "error": "RDKit not available"}
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"score": 0, "components": {}, "error": "Invalid SMILES"}
            
            components = {}
            
            # SA score (1-10 scale, lower is easier)
            sa_raw = self._calculate_sa_score(mol)
            sa_score = max(0, (10 - sa_raw) / 9 * 100)
            components["sa_score"] = round(sa_score, 1)
            
            # Complexity
            try:
                complexity = GraphDescriptors.BertzCT(mol)
                norm_complexity = min(1.0, complexity / 500)
                components["complexity"] = round((1 - norm_complexity) * 100, 1)
            except Exception:
                components["complexity"] = 70.0
            
            # Starting materials
            num_carbon = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
            starting = 80.0
            if num_carbon > 25:
                starting -= 15
            unusual = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ["P", "Si", "B", "Se"])
            starting -= unusual * 20
            components["starting_materials"] = max(0, min(100, starting))
            
            # Steps estimate
            steps = 1 + max(0, mol.GetRingInfo().NumRings() - 2)
            components["estimated_steps"] = steps
            
            overall = (components["sa_score"] * 0.4 + components["complexity"] * 0.3 +
                      components["starting_materials"] * 0.2 + max(0, 100 - steps * 15) * 0.1)
            
            return {"score": round(overall, 1), "components": components}
        except Exception as e:
            logger.error(f"SA scoring error: {e}")
            return {"score": 50, "components": {}, "error": str(e)}
    
    def _calculate_sa_score(self, mol) -> float:
        try:
            score = 2.0
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            score += min(num_rings * 0.3, 2.0)
            chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            score += chiral * 0.5
            num_atoms = mol.GetNumHeavyAtoms()
            if num_atoms > 35:
                score += (num_atoms - 35) * 0.1
            return min(10.0, max(1.0, score))
        except Exception:
            return 5.0
