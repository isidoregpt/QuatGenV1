"""Environmental Scoring - Predict environmental impact using ADMET models and RDKit"""

import logging
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from scoring.admet_models import ADMETPredictor

logger = logging.getLogger(__name__)


class EnvironmentalScorer:
    """
    Environmental impact scorer using ADMET models and RDKit descriptors.

    Uses solubility and lipophilicity predictions from ADMET models
    to assess environmental fate, with RDKit heuristics as fallback.
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
        Initialize the environmental scorer.

        Args:
            admet_predictor: Optional shared ADMETPredictor instance
        """
        self.admet_predictor = admet_predictor
        self._is_ready = True

        if self.uses_admet:
            logger.info("Environmental scorer initialized with ADMET models")
        else:
            logger.info("Environmental scorer initialized (RDKit heuristics mode)")

    async def score(self, smiles: str) -> Dict:
        """
        Score a molecule for environmental impact.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Dictionary with overall score and component scores
        """
        components = {}
        raw_values = {}

        # Try ADMET predictions first for relevant properties
        if self.uses_admet:
            admet_components, admet_raw = self._score_with_admet(smiles)
            components.update(admet_components)
            raw_values.update(admet_raw)

        # Fill in with RDKit heuristics for remaining components
        rdkit_components = self._score_with_rdkit(smiles)
        for key, value in rdkit_components.items():
            if key not in components:
                components[key] = value

        # Calculate overall environmental score
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
        Score using ADMET models for environmental-relevant properties.

        Returns:
            Tuple of (components dict, raw_values dict)
        """
        components = {}
        raw_values = {}

        available = self.admet_predictor.available_properties

        # Solubility affects environmental fate
        if "solubility" in available:
            result = self.admet_predictor.predict(smiles, "solubility")
            if "value" in result:
                sol_log = result["value"]  # log mol/L
                raw_values["solubility_log_mol_L"] = sol_log

                # Higher solubility generally better for biodegradation
                # but very high solubility can spread more easily
                # Optimal range: -3 to -1 log mol/L
                if -3 <= sol_log <= -1:
                    sol_score = 80
                elif sol_log > -1:
                    # Very soluble - spreads easily
                    sol_score = 70 - (sol_log + 1) * 10
                else:
                    # Poor solubility - may accumulate
                    sol_score = 60 + (sol_log + 3) * 10

                components["solubility_score"] = min(100, max(0, sol_score))

        # Lipophilicity affects bioaccumulation potential
        if "lipophilicity" in available:
            result = self.admet_predictor.predict(smiles, "lipophilicity")
            if "value" in result:
                logd = result["value"]
                raw_values["lipophilicity_logD"] = logd

                # Lower logD = less bioaccumulation = better environmental profile
                # logD < 1: minimal bioaccumulation risk
                # logD 1-3: moderate risk
                # logD > 3: high bioaccumulation risk
                if logd < 1:
                    bioacc = 95
                elif logd < 3:
                    bioacc = 85 - (logd - 1) * 15
                else:
                    bioacc = max(20, 55 - (logd - 3) * 15)

                components["bioaccumulation"] = bioacc

        # Clearance can indicate how fast compound is metabolized/degraded
        if "clearance" in available:
            result = self.admet_predictor.predict(smiles, "clearance")
            if "value" in result:
                clearance = result["value"]
                raw_values["metabolic_clearance"] = clearance

                # Higher clearance suggests faster degradation potential
                # but this is for mammalian systems, not environmental
                # Use as a proxy for general degradability
                if clearance > 50:
                    clear_score = 80
                elif clearance > 20:
                    clear_score = 65
                else:
                    clear_score = 50

                components["degradability_proxy"] = clear_score

        return components, raw_values

    def _score_with_rdkit(self, smiles: str) -> Dict:
        """
        Score using RDKit-based heuristics.

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
            num_aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())

            components = {}

            # Biodegradability estimate
            if "biodegradability" not in components:
                biodeg = 50.0
                # Moderate logP and MW generally better for biodegradation
                if 2 <= logp <= 5:
                    biodeg += 15
                if mw < 400:
                    biodeg += 10
                # Aromatic rings reduce biodegradability
                biodeg -= num_aromatic * 3
                components["biodegradability"] = max(0, min(100, biodeg))

            # Aquatic toxicity safety (if not covered by ADMET)
            if "aquatic_toxicity" not in components:
                # Higher logP generally means more toxic to aquatic life
                if logp < 2:
                    aquatic = 80
                elif logp < 4:
                    aquatic = 60 - (logp - 2) * 10
                else:
                    aquatic = max(20, 40 - (logp - 4) * 10)
                components["aquatic_toxicity"] = aquatic

            # Bioaccumulation (if not covered by ADMET)
            if "bioaccumulation" not in components:
                # Estimate BCF from logP
                log_bcf = 0.85 * logp - 0.70
                if log_bcf < 1:
                    bcf = 95
                elif log_bcf < 2:
                    bcf = 85
                elif log_bcf < 3:
                    bcf = 65
                else:
                    bcf = 40
                components["bioaccumulation"] = bcf

            # Persistence in environment
            if "persistence" not in components:
                persist = 70
                # Aromatic rings increase persistence
                persist -= num_aromatic * 10
                # High logP increases persistence
                if logp > 5:
                    persist -= (logp - 5) * 8
                # Halogens increase persistence
                halogens = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ["F", "Cl", "Br", "I"])
                persist -= halogens * 5
                components["persistence"] = max(0, min(100, persist))

            return components

        except Exception as e:
            logger.error(f"RDKit environmental scoring error: {e}")
            return {}

    def _calculate_overall_score(self, components: Dict) -> float:
        """
        Calculate weighted overall environmental score.

        Args:
            components: Dictionary of component scores

        Returns:
            Overall environmental score (0-100)
        """
        # Weights for different environmental aspects
        weights = {
            # ADMET-derived components
            "solubility_score": 0.15,
            "degradability_proxy": 0.10,
            # RDKit and ADMET shared components
            "bioaccumulation": 0.25,
            "biodegradability": 0.20,
            "aquatic_toxicity": 0.20,
            "persistence": 0.10
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
