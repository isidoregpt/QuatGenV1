"""Safety Scoring - Predict human toxicity using ADMET models and RDKit fallbacks"""

import logging
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from scoring.admet_models import ADMETPredictor

logger = logging.getLogger(__name__)


class SafetyScorer:
    """
    Safety scorer that predicts human toxicity using ADMET models.

    Uses ChemFM ADMET models for toxicity prediction when available,
    with RDKit-based heuristics as fallback.
    """

    def __init__(self):
        self._is_ready = False
        self.admet_predictor: Optional["ADMETPredictor"] = None

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def uses_admet(self) -> bool:
        return self.admet_predictor is not None and self.admet_predictor.is_ready

    async def initialize(self, admet_predictor: Optional["ADMETPredictor"] = None):
        """
        Initialize the safety scorer.

        Args:
            admet_predictor: Optional shared ADMETPredictor instance
        """
        self.admet_predictor = admet_predictor
        self._is_ready = True

        if self.uses_admet:
            logger.info("Safety scorer initialized with ADMET models")
        else:
            logger.info("Safety scorer initialized (RDKit heuristics mode)")

    async def score(self, smiles: str) -> Dict:
        """
        Score a molecule for human safety.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Dictionary with overall score and component scores
        """
        components = {}
        raw_values = {}

        # Try ADMET predictions first
        if self.uses_admet:
            admet_components, admet_raw = self._score_with_admet(smiles)
            components.update(admet_components)
            raw_values.update(admet_raw)

        # Fill in with RDKit heuristics for any missing components
        rdkit_components = self._score_with_rdkit(smiles)
        for key, value in rdkit_components.items():
            if key not in components:
                components[key] = value

        # Calculate overall safety score
        overall = self._calculate_overall_score(components)

        result = {
            "score": round(overall, 1),
            "components": components,
            "used_admet": self.uses_admet
        }

        # Include raw ADMET values for detailed analysis
        if raw_values:
            result["raw_predictions"] = raw_values

        return result

    def _score_with_admet(self, smiles: str) -> tuple:
        """
        Score using ADMET models.

        Returns:
            Tuple of (components dict, raw_values dict)
        """
        components = {}
        raw_values = {}

        available = self.admet_predictor.available_properties

        # LD50 - Acute oral toxicity (higher LD50 = safer)
        if "ld50" in available:
            result = self.admet_predictor.predict(smiles, "ld50")
            if "value" in result:
                ld50_log = result["value"]
                raw_values["ld50_log_mg_kg"] = ld50_log
                # Convert log LD50 to safety score (0-100)
                # Higher LD50 = less toxic = higher score
                # log(LD50) of 2 (100 mg/kg) = moderately toxic = score ~50
                # log(LD50) of 4 (10000 mg/kg) = relatively safe = score ~100
                components["acute_toxicity"] = min(100, max(0, (ld50_log + 1) * 25))

        # hERG - Cardiac toxicity (lower probability of inhibition = safer)
        if "herg" in available:
            result = self.admet_predictor.predict(smiles, "herg")
            if "probability" in result:
                herg_prob = result["probability"]
                raw_values["herg_inhibition_prob"] = herg_prob
                # Invert: high probability of hERG inhibition = unsafe
                components["cardiac_safety"] = (1 - herg_prob) * 100

        # Ames - Mutagenicity (lower probability = safer)
        if "ames" in available:
            result = self.admet_predictor.predict(smiles, "ames")
            if "probability" in result:
                ames_prob = result["probability"]
                raw_values["ames_positive_prob"] = ames_prob
                components["mutagenicity_safety"] = (1 - ames_prob) * 100

        # DILI - Liver toxicity (lower probability = safer)
        if "dili" in available:
            result = self.admet_predictor.predict(smiles, "dili")
            if "probability" in result:
                dili_prob = result["probability"]
                raw_values["dili_risk_prob"] = dili_prob
                components["liver_safety"] = (1 - dili_prob) * 100

        # BBB - Blood-brain barrier (for CNS toxicity consideration)
        if "bbb" in available:
            result = self.admet_predictor.predict(smiles, "bbb")
            if "probability" in result:
                bbb_prob = result["probability"]
                raw_values["bbb_penetration_prob"] = bbb_prob
                # For quaternary ammonium compounds (disinfectants),
                # lower BBB penetration is generally safer
                components["cns_safety"] = (1 - bbb_prob) * 100

        return components, raw_values

    def _score_with_rdkit(self, smiles: str) -> Dict:
        """
        Score using RDKit-based heuristics as fallback.

        Returns:
            Dictionary of component scores
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}

            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)

            components = {}

            # Oral toxicity estimate - lower logP generally safer
            if "acute_toxicity" not in components:
                if logp < 2:
                    oral = 90
                elif logp < 4:
                    oral = 70
                else:
                    oral = max(30, 60 - (logp - 4) * 10)
                components["acute_toxicity"] = oral

            # Skin irritation - based on logP and charge
            if "skin_irritation" not in components:
                if logp < 1:
                    skin = 95
                elif logp < 3:
                    skin = 75
                else:
                    skin = max(35, 65 - (logp - 3) * 10)
                components["skin_irritation"] = skin

            # Eye irritation - quats are generally irritants
            if "eye_irritation" not in components:
                charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
                if charge == 0:
                    eye = 90
                elif charge == 1:
                    eye = 60
                else:
                    eye = 40
                components["eye_irritation"] = eye

            # Respiratory safety - higher MW safer (less volatile)
            if "respiratory" not in components:
                if mw > 400:
                    resp = 90
                elif mw > 300:
                    resp = 75
                else:
                    resp = 60
                components["respiratory"] = resp

            return components

        except Exception as e:
            logger.error(f"RDKit safety scoring error: {e}")
            return {}

    def _calculate_overall_score(self, components: Dict) -> float:
        """
        Calculate weighted overall safety score.

        Args:
            components: Dictionary of component scores

        Returns:
            Overall safety score (0-100)
        """
        # Weights for different safety aspects
        # ADMET-based components have higher weights when available
        weights = {
            # ADMET model components (primary)
            "acute_toxicity": 0.25,
            "cardiac_safety": 0.20,
            "mutagenicity_safety": 0.20,
            "liver_safety": 0.15,
            "cns_safety": 0.05,
            # RDKit heuristic components (secondary/fallback)
            "skin_irritation": 0.10,
            "eye_irritation": 0.10,
            "respiratory": 0.05
        }

        total_weight = 0
        weighted_sum = 0

        for key, weight in weights.items():
            if key in components:
                weighted_sum += components[key] * weight
                total_weight += weight

        if total_weight == 0:
            return 50.0  # Default neutral score

        return weighted_sum / total_weight
