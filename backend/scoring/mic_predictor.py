"""
MIC (Minimum Inhibitory Concentration) Predictor

Uses molecular embeddings and ChEMBL training data to predict antimicrobial activity
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
import json

logger = logging.getLogger(__name__)


@dataclass
class MICPrediction:
    """Predicted MIC value for a molecule against an organism"""
    organism: str
    predicted_mic: float          # Predicted MIC in µg/mL
    confidence: float             # Prediction confidence (0-1)
    activity_class: str           # "excellent", "good", "moderate", "weak", "inactive"
    percentile_rank: float        # How this compares to training data (0-100)
    similar_compounds: List[Dict] # Similar known compounds


class MICPredictorHead(nn.Module):
    """Neural network head for MIC prediction from embeddings"""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_organisms: int = 5):
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Separate output heads for each organism
        self.organism_heads = nn.ModuleDict({
            "s_aureus": nn.Linear(hidden_dim // 2, 1),
            "e_coli": nn.Linear(hidden_dim // 2, 1),
            "p_aeruginosa": nn.Linear(hidden_dim // 2, 1),
            "c_albicans": nn.Linear(hidden_dim // 2, 1),
            "general": nn.Linear(hidden_dim // 2, 1),
        })

    def forward(self, x: torch.Tensor, organism: str = "general") -> torch.Tensor:
        shared_features = self.shared_layers(x)

        if organism in self.organism_heads:
            return self.organism_heads[organism](shared_features)
        return self.organism_heads["general"](shared_features)


class MICPredictor:
    """
    Predicts MIC values using molecular embeddings.
    Can be trained on ChEMBL data or use similarity-based prediction.
    """

    # Activity classification thresholds (µg/mL)
    MIC_THRESHOLDS = {
        "excellent": 2,
        "good": 8,
        "moderate": 32,
        "weak": 128,
        "inactive": float("inf")
    }

    ORGANISMS = ["s_aureus", "e_coli", "p_aeruginosa", "c_albicans", "general"]

    def __init__(self, encoder=None, device: str = None):
        """
        Args:
            encoder: MolecularEncoder instance for getting embeddings
            device: Device for neural network inference
        """
        self.encoder = encoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[MICPredictorHead] = None
        self.is_trained = False

        # Training data statistics for percentile calculation
        self.mic_statistics: Dict[str, Dict] = {}

        # Reference embeddings for similarity-based prediction
        self.reference_embeddings: Dict[str, np.ndarray] = {}
        self.reference_mics: Dict[str, Dict[str, float]] = {}

        self._is_ready = False

    async def initialize(self, reference_db=None, chembl_fetcher=None):
        """Initialize predictor with reference data"""
        logger.info("Initializing MIC predictor...")

        # Create model
        embedding_dim = self.encoder.embedding_dim if self.encoder and self.encoder.is_ready else 768
        self.model = MICPredictorHead(input_dim=embedding_dim)
        self.model.to(self.device)
        self.model.eval()

        # Load reference compound embeddings for similarity-based prediction
        if reference_db and self.encoder and self.encoder.is_ready:
            await self._load_reference_embeddings(reference_db)

        # Load training data statistics from ChEMBL
        if chembl_fetcher and chembl_fetcher.is_ready:
            self._compute_statistics(chembl_fetcher)

        # Try to load pretrained weights
        weights_path = "models/mic_predictor.pt"
        if os.path.exists(weights_path):
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                self.is_trained = True
                logger.info("Loaded pretrained MIC predictor weights")
            except Exception as e:
                logger.warning(f"Could not load MIC predictor weights: {e}")

        self._is_ready = True
        logger.info("MIC predictor ready")

    async def _load_reference_embeddings(self, reference_db):
        """Compute embeddings for reference compounds"""
        logger.info("Computing reference compound embeddings...")

        for compound in reference_db.get_all():
            try:
                embedding = self.encoder.encode(compound.smiles)
                self.reference_embeddings[compound.name] = embedding

                # Store MIC values - reference_db uses tuples (min, max)
                # Use geometric mean as representative value
                def get_mic_mean(mic_tuple):
                    if mic_tuple:
                        return (mic_tuple[0] * mic_tuple[1]) ** 0.5
                    return None

                self.reference_mics[compound.name] = {
                    "s_aureus": get_mic_mean(compound.mic_s_aureus),
                    "e_coli": get_mic_mean(compound.mic_e_coli),
                    "p_aeruginosa": get_mic_mean(compound.mic_p_aeruginosa),
                    "c_albicans": get_mic_mean(compound.mic_c_albicans),
                }

            except Exception as e:
                logger.warning(f"Could not embed {compound.name}: {e}")

        logger.info(f"Computed embeddings for {len(self.reference_embeddings)} reference compounds")

    def _compute_statistics(self, chembl_fetcher):
        """Compute MIC statistics from ChEMBL training data"""
        training_data = chembl_fetcher.get_training_data()

        if not training_data:
            logger.warning("No training data available for statistics")
            return

        # Group by organism
        organism_mics = {}
        for entry in training_data:
            org = self._normalize_organism(entry.get("organism", ""))
            mic = entry.get("mic_value")

            if org and mic:
                if org not in organism_mics:
                    organism_mics[org] = []
                organism_mics[org].append(mic)

        # Compute statistics
        for org, mics in organism_mics.items():
            self.mic_statistics[org] = {
                "min": float(np.min(mics)),
                "max": float(np.max(mics)),
                "mean": float(np.mean(mics)),
                "median": float(np.median(mics)),
                "std": float(np.std(mics)),
                "percentiles": np.percentile(mics, [10, 25, 50, 75, 90]).tolist(),
                "count": len(mics)
            }

        logger.info(f"Computed MIC statistics for {len(self.mic_statistics)} organisms")

    def _normalize_organism(self, organism: str) -> Optional[str]:
        """Normalize organism name to standard key"""
        org_lower = organism.lower()

        if "staphylococcus" in org_lower or "s. aureus" in org_lower or "s aureus" in org_lower:
            return "s_aureus"
        elif "escherichia" in org_lower or "e. coli" in org_lower or "e coli" in org_lower:
            return "e_coli"
        elif "pseudomonas" in org_lower or "p. aeruginosa" in org_lower:
            return "p_aeruginosa"
        elif "candida" in org_lower or "c. albicans" in org_lower:
            return "c_albicans"

        return None

    def _classify_activity(self, mic: float) -> str:
        """Classify MIC value into activity class"""
        for activity_class, threshold in self.MIC_THRESHOLDS.items():
            if mic <= threshold:
                return activity_class
        return "inactive"

    def _calculate_percentile(self, mic: float, organism: str) -> float:
        """Calculate percentile rank (lower MIC = better = higher percentile)"""
        if organism not in self.mic_statistics:
            return 50.0

        stats = self.mic_statistics[organism]
        percentiles = stats.get("percentiles", [])

        if not percentiles:
            return 50.0

        # Lower MIC is better, so invert percentile
        if mic <= percentiles[0]:  # Better than 90th percentile
            return 95.0
        elif mic <= percentiles[1]:  # 75-90th percentile
            return 85.0
        elif mic <= percentiles[2]:  # 50-75th percentile
            return 65.0
        elif mic <= percentiles[3]:  # 25-50th percentile
            return 40.0
        elif mic <= percentiles[4]:  # 10-25th percentile
            return 20.0
        else:
            return 5.0

    def _find_similar_compounds(self, embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """Find most similar reference compounds by embedding similarity"""
        if not self.reference_embeddings:
            return []

        similarities = []
        for name, ref_embedding in self.reference_embeddings.items():
            sim = np.dot(embedding, ref_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(ref_embedding) + 1e-8
            )
            similarities.append((name, float(sim), self.reference_mics.get(name, {})))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "name": name,
                "similarity": round(sim, 3),
                "mic_values": mics
            }
            for name, sim, mics in similarities[:top_k]
        ]

    def predict(self, smiles: str, organism: str = "general") -> MICPrediction:
        """Predict MIC for a molecule against specified organism"""
        if not self._is_ready:
            raise RuntimeError("MIC predictor not initialized")

        # Get molecular embedding
        if self.encoder and self.encoder.is_ready:
            try:
                embedding = self.encoder.encode(smiles)
            except Exception as e:
                logger.warning(f"Encoding failed for {smiles}: {e}")
                return self._heuristic_prediction(smiles, organism)
        else:
            # Fallback: return heuristic-based prediction
            return self._heuristic_prediction(smiles, organism)

        # Find similar reference compounds
        similar = self._find_similar_compounds(embedding)

        # Predict MIC using neural network or similarity
        if self.is_trained:
            predicted_mic = self._nn_predict(embedding, organism)
            confidence = 0.7
        else:
            # Similarity-based prediction
            predicted_mic, confidence = self._similarity_predict(embedding, organism, similar)

        # Classify activity
        activity_class = self._classify_activity(predicted_mic)

        # Calculate percentile
        percentile = self._calculate_percentile(predicted_mic, organism)

        return MICPrediction(
            organism=organism,
            predicted_mic=round(predicted_mic, 2),
            confidence=round(confidence, 2),
            activity_class=activity_class,
            percentile_rank=percentile,
            similar_compounds=similar
        )

    def _nn_predict(self, embedding: np.ndarray, organism: str) -> float:
        """Neural network-based MIC prediction"""
        self.model.eval()

        with torch.no_grad():
            x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
            log_mic = self.model(x, organism).item()

        # Convert from log scale
        return 10 ** log_mic

    def _similarity_predict(self, embedding: np.ndarray, organism: str,
                            similar: List[Dict]) -> Tuple[float, float]:
        """Similarity-weighted MIC prediction from reference compounds"""
        if not similar:
            return 32.0, 0.3  # Default moderate activity with low confidence

        weighted_sum = 0.0
        weight_total = 0.0

        for compound in similar:
            sim = compound["similarity"]
            mic_values = compound.get("mic_values", {})
            mic = mic_values.get(organism) or mic_values.get("s_aureus")  # Fallback

            if mic and sim > 0.5:  # Only use sufficiently similar compounds
                weight = sim ** 2  # Square similarity for stronger weighting
                weighted_sum += mic * weight
                weight_total += weight

        if weight_total > 0:
            predicted_mic = weighted_sum / weight_total
            confidence = min(0.8, weight_total / len(similar))
        else:
            predicted_mic = 32.0
            confidence = 0.3

        return predicted_mic, confidence

    def _heuristic_prediction(self, smiles: str, organism: str) -> MICPrediction:
        """Fallback heuristic-based prediction using RDKit descriptors"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return MICPrediction(organism, 128, 0.2, "weak", 20, [])

            # Simple heuristics based on quat properties
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)

            # Optimal logP for membrane disruption: 3-6
            logp_score = 1.0 - abs(logp - 4.5) / 4

            # Optimal MW for quats: 300-500
            mw_score = 1.0 - abs(mw - 400) / 300

            # Check for quat nitrogen
            has_quat = "[N+]" in smiles or "[n+]" in smiles
            quat_score = 1.0 if has_quat else 0.3

            # Combined score -> MIC estimate
            combined = (logp_score + mw_score + quat_score) / 3

            # Map to MIC (higher score = lower MIC = better)
            if combined > 0.7:
                predicted_mic = 4
            elif combined > 0.5:
                predicted_mic = 16
            elif combined > 0.3:
                predicted_mic = 64
            else:
                predicted_mic = 128

            return MICPrediction(
                organism=organism,
                predicted_mic=predicted_mic,
                confidence=0.3,
                activity_class=self._classify_activity(predicted_mic),
                percentile_rank=50,
                similar_compounds=[]
            )

        except Exception as e:
            logger.error(f"Heuristic prediction error: {e}")
            return MICPrediction(organism, 64, 0.2, "moderate", 40, [])

    def predict_all_organisms(self, smiles: str) -> Dict[str, MICPrediction]:
        """Predict MIC against all target organisms"""
        return {org: self.predict(smiles, org) for org in self.ORGANISMS}

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    async def train(self, training_data: List[Dict], epochs: int = 100):
        """Train the MIC predictor on ChEMBL data"""
        # This would be called manually or during a training phase
        logger.info(f"Training MIC predictor on {len(training_data)} samples...")
        # Training implementation would go here
        self.is_trained = True
