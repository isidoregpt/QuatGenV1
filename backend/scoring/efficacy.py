"""
Efficacy Scoring - Predict antimicrobial activity using MIC predictor
"""

import logging
from typing import Dict, Optional

from scoring.mic_predictor import MICPredictor, MICPrediction

logger = logging.getLogger(__name__)

# Try to import RDKit for fallback calculations
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available - efficacy scoring will be limited")


class EfficacyScorer:
    """
    Scores molecules for antimicrobial efficacy.
    Uses MIC predictions when available, falls back to QSAR heuristics.
    """

    def __init__(self):
        self._is_ready = False
        self.mic_predictor: Optional[MICPredictor] = None
        self.encoder = None

    async def initialize(self, encoder=None, reference_db=None, chembl_fetcher=None, **kwargs):
        """
        Initialize efficacy scorer with optional MIC predictor.

        Args:
            encoder: MolecularEncoder for embeddings
            reference_db: ReferenceDatabase with known quats
            chembl_fetcher: ChEMBLFetcher with training data
        """
        self.encoder = encoder

        # Initialize MIC predictor if encoder is available
        if encoder:
            try:
                self.mic_predictor = MICPredictor(encoder=encoder)
                await self.mic_predictor.initialize(
                    reference_db=reference_db,
                    chembl_fetcher=chembl_fetcher
                )
                logger.info("MIC predictor initialized for efficacy scoring")
            except Exception as e:
                logger.warning(f"Could not initialize MIC predictor: {e}")
                self.mic_predictor = None

        self._is_ready = True
        logger.info("Efficacy scorer initialized")

    async def score(self, smiles: str) -> Dict:
        """
        Score a molecule for antimicrobial efficacy.

        Returns dict with:
            - score: Overall efficacy score (0-100)
            - components: Individual scoring components
            - mic_predictions: Predicted MIC values per organism (if available)
        """
        components = {}
        mic_predictions = {}

        # Get MIC predictions if predictor is available
        if self.mic_predictor and self.mic_predictor.is_ready:
            mic_results = self._score_with_mic_predictor(smiles)
            components.update(mic_results["components"])
            mic_predictions = mic_results["predictions"]

        # RDKit-based QSAR components (always calculated as additional features)
        if RDKIT_AVAILABLE:
            qsar_results = self._score_with_qsar(smiles)
            # Only add QSAR components if MIC predictor didn't provide them
            for key, value in qsar_results.items():
                if key not in components:
                    components[key] = value

        # Calculate overall score
        overall = self._calculate_overall_score(components)

        result = {
            "score": round(overall, 1),
            "components": components,
            "used_mic_predictor": self.mic_predictor is not None and self.mic_predictor.is_ready
        }

        if mic_predictions:
            result["mic_predictions"] = mic_predictions

        # Add molecular properties
        if RDKIT_AVAILABLE:
            props = self._get_molecular_properties(smiles)
            result.update(props)

        return result

    def _score_with_mic_predictor(self, smiles: str) -> Dict:
        """Get efficacy scores from MIC predictions"""
        import math

        components = {}
        predictions = {}

        # Predict MIC for key organisms
        organisms = {
            "s_aureus": ("gram_positive_mic", 0.30),
            "e_coli": ("gram_negative_mic", 0.25),
            "p_aeruginosa": ("pseudomonas_mic", 0.15),
            "c_albicans": ("antifungal", 0.20),
            "general": ("general_antimicrobial", 0.10),
        }

        for org, (component_name, weight) in organisms.items():
            try:
                prediction = self.mic_predictor.predict(smiles, org)
                predictions[org] = {
                    "mic": prediction.predicted_mic,
                    "confidence": prediction.confidence,
                    "activity_class": prediction.activity_class,
                    "percentile": prediction.percentile_rank,
                    "similar_compounds": prediction.similar_compounds[:2]  # Top 2 similar
                }

                # Convert MIC to score (lower MIC = higher score)
                # MIC 1 µg/mL -> 100, MIC 128 µg/mL -> 0
                mic = prediction.predicted_mic
                if mic <= 1:
                    score = 100
                elif mic >= 128:
                    score = 0
                else:
                    # Logarithmic scale: score = 100 - (log2(MIC) * 14.3)
                    score = max(0, 100 - (math.log2(mic) * 14.3))

                components[component_name] = round(score, 1)

            except Exception as e:
                logger.warning(f"MIC prediction failed for {org}: {e}")

        return {"components": components, "predictions": predictions}

    def _score_with_qsar(self, smiles: str) -> Dict:
        """Calculate QSAR-based efficacy components"""
        components = {}

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"qsar_error": "Invalid SMILES"}

            # Molecular properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)

            # Count carbons in longest chain (proxy for alkyl chain length)
            chain_carbons = sum(1 for a in mol.GetAtoms()
                                if a.GetSymbol() == "C" and not a.GetIsAromatic() and not a.IsInRing())

            # Check for quaternary nitrogen
            formal_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
            has_quat = "[N+]" in smiles or "[n+]" in smiles

            # CMC score - Critical Micelle Concentration
            # Optimal chain length for surfactants: C12-C16
            optimal_chain = 14
            chain_score = max(0, 100 - abs(chain_carbons - optimal_chain) * 8)
            components["cmc_score"] = round(chain_score, 1)

            # Membrane disruption score
            # Requires positive charge and appropriate lipophilicity
            if has_quat and 2 <= logp <= 6:
                membrane_score = 80 + (1 - abs(logp - 4) / 2) * 20
            elif has_quat:
                membrane_score = 50 + (1 - abs(logp - 4) / 4) * 30
            else:
                membrane_score = 30
            components["membrane_disruption"] = round(membrane_score, 1)

            # Hydrophile-lipophile balance proxy
            hlb_score = 100 - abs(logp - 4) * 15
            components["hlb_score"] = round(max(0, min(100, hlb_score)), 1)

        except Exception as e:
            logger.error(f"QSAR scoring error: {e}")
            components["qsar_error"] = str(e)

        return components

    def _calculate_overall_score(self, components: Dict) -> float:
        """Calculate weighted overall efficacy score"""
        weights = {
            # MIC-based components (primary)
            "gram_positive_mic": 0.25,
            "gram_negative_mic": 0.20,
            "antifungal": 0.15,
            "pseudomonas_mic": 0.10,
            "general_antimicrobial": 0.05,
            # QSAR-based components (secondary)
            "cmc_score": 0.10,
            "membrane_disruption": 0.10,
            "hlb_score": 0.05,
        }

        total_weight = 0
        weighted_sum = 0

        for key, weight in weights.items():
            if key in components and isinstance(components[key], (int, float)):
                weighted_sum += components[key] * weight
                total_weight += weight

        if total_weight == 0:
            return 50.0  # Default neutral score

        return weighted_sum / total_weight

    def _get_molecular_properties(self, smiles: str) -> Dict:
        """Extract molecular properties for the response"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}

            return {
                "mw": round(Descriptors.MolWt(mol), 1),
                "logp": round(Descriptors.MolLogP(mol), 2),
                "tpsa": round(Descriptors.TPSA(mol), 1),
                "hbd": rdMolDescriptors.CalcNumHBD(mol),
                "hba": rdMolDescriptors.CalcNumHBA(mol),
                "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                "chain_length": sum(1 for a in mol.GetAtoms()
                                   if a.GetSymbol() == "C" and not a.GetIsAromatic())
            }
        except Exception:
            return {}

    @property
    def is_ready(self) -> bool:
        return self._is_ready
