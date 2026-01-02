"""Scoring Pipeline - Multi-objective scoring for quaternary ammonium compounds"""

import asyncio
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    use_gpu: bool = True
    batch_size: int = 32
    cache_scores: bool = True
    # Molecular encoder settings
    use_molecular_encoder: bool = True
    encoder_model_name: str = "DeepChem/ChemBERTa-77M-MLM"
    encoder_pooling: str = "mean"  # "mean" or "cls"
    encoder_device: Optional[str] = None  # Auto-detect if None
    # ADMET models settings
    use_admet_models: bool = True
    admet_models: Optional[List[str]] = None  # None = load all available
    admet_lazy_load: bool = True  # Load models on first use
    # MIC predictor settings
    use_mic_predictor: bool = True


class ScoringPipeline:
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()
        self.efficacy_scorer = None
        self.safety_scorer = None
        self.environmental_scorer = None
        self.sa_scorer = None
        self.molecular_encoder = None
        self.admet_predictor = None
        self._is_ready = False
        # Data sources for MIC predictor
        self.reference_db = None
        self.chembl_fetcher = None

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def encoder_ready(self) -> bool:
        return self.molecular_encoder is not None and self.molecular_encoder.is_ready

    @property
    def admet_ready(self) -> bool:
        return self.admet_predictor is not None and self.admet_predictor.is_ready

    @property
    def embedding_dim(self) -> int:
        if self.molecular_encoder and self.molecular_encoder.is_ready:
            return self.molecular_encoder.embedding_dim
        return 0

    @property
    def mic_predictor_ready(self) -> bool:
        return (self.efficacy_scorer is not None and
                self.efficacy_scorer.mic_predictor is not None and
                self.efficacy_scorer.mic_predictor.is_ready)

    async def initialize(self):
        logger.info("Initializing scoring pipeline...")
        from scoring.efficacy import EfficacyScorer
        from scoring.safety import SafetyScorer
        from scoring.environmental import EnvironmentalScorer
        from scoring.sa_score import SAScorer

        # Initialize molecular encoder FIRST (needed for MIC predictor in efficacy scorer)
        if self.config.use_molecular_encoder:
            try:
                from scoring.molecular_encoder import MolecularEncoder
                logger.info(f"Loading molecular encoder: {self.config.encoder_model_name}")
                self.molecular_encoder = MolecularEncoder(
                    model_name=self.config.encoder_model_name,
                    device=self.config.encoder_device,
                    pooling=self.config.encoder_pooling
                )
                success = await self.molecular_encoder.initialize()
                if success:
                    logger.info(f"Molecular encoder ready (dim={self.molecular_encoder.embedding_dim})")
                else:
                    logger.warning("Molecular encoder failed to load, continuing without embeddings")
                    self.molecular_encoder = None
            except Exception as e:
                logger.warning(f"Failed to initialize molecular encoder: {e}")
                self.molecular_encoder = None

        # Initialize ADMET predictor (shared resource for safety/environmental)
        if self.config.use_admet_models:
            try:
                from scoring.admet_models import ADMETPredictor
                logger.info("Initializing ADMET predictor...")
                self.admet_predictor = ADMETPredictor(
                    models_to_load=self.config.admet_models,
                    lazy_load=self.config.admet_lazy_load
                )
                success = await self.admet_predictor.initialize()
                if success:
                    logger.info(f"ADMET predictor ready with {len(self.admet_predictor.available_properties)} models")
                else:
                    logger.warning("ADMET predictor failed to initialize, using RDKit fallback")
                    self.admet_predictor = None
            except Exception as e:
                logger.warning(f"Failed to initialize ADMET predictor: {e}")
                self.admet_predictor = None

        # Initialize efficacy scorer with MIC predictor (uses encoder, reference_db, chembl_fetcher)
        self.efficacy_scorer = EfficacyScorer()
        await self.efficacy_scorer.initialize(
            encoder=self.molecular_encoder,
            reference_db=self.reference_db,
            chembl_fetcher=self.chembl_fetcher
        )

        # Initialize other scorers with ADMET predictor
        self.safety_scorer = SafetyScorer()
        await self.safety_scorer.initialize(admet_predictor=self.admet_predictor)

        self.environmental_scorer = EnvironmentalScorer()
        await self.environmental_scorer.initialize(admet_predictor=self.admet_predictor)

        self.sa_scorer = SAScorer()
        await self.sa_scorer.initialize()

        self._is_ready = True
        logger.info("Scoring pipeline ready")
    
    async def score_molecule(self, smiles: str, include_embedding: bool = False) -> Dict:
        if not self._is_ready:
            raise RuntimeError("Scoring pipeline not initialized")

        efficacy_result, safety_result, env_result, sa_result = await asyncio.gather(
            self.efficacy_scorer.score(smiles),
            self.safety_scorer.score(smiles),
            self.environmental_scorer.score(smiles),
            self.sa_scorer.score(smiles)
        )

        result = {
            "efficacy": efficacy_result["score"],
            "safety": safety_result["score"],
            "environmental": env_result["score"],
            "sa_score": sa_result["score"],
            "mw": efficacy_result.get("mw"),
            "logp": efficacy_result.get("logp"),
            "chain_length": efficacy_result.get("chain_length")
        }

        # Include embedding if requested and encoder is available
        if include_embedding and self.encoder_ready:
            try:
                embedding = self.molecular_encoder.encode(smiles)
                result["embedding"] = embedding.tolist()
                result["embedding_dim"] = len(embedding)
            except Exception as e:
                logger.warning(f"Failed to compute embedding for {smiles}: {e}")
                result["embedding"] = None
                result["embedding_dim"] = 0

        return result

    async def score_batch(self, smiles_list: List[str]) -> List[Dict]:
        return await asyncio.gather(
            *[self.score_molecule(s) for s in smiles_list],
            return_exceptions=True
        )

    async def get_embedding(self, smiles: str) -> Optional[np.ndarray]:
        """
        Get the molecular embedding for a single SMILES string.

        Args:
            smiles: SMILES string to encode

        Returns:
            numpy array of embedding, or None if encoder not available
        """
        if not self.encoder_ready:
            logger.warning("Molecular encoder not available")
            return None

        try:
            return self.molecular_encoder.encode(smiles)
        except Exception as e:
            logger.error(f"Failed to encode SMILES: {e}")
            return None

    async def get_embeddings_batch(self, smiles_list: List[str]) -> Optional[np.ndarray]:
        """
        Get molecular embeddings for a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings to encode

        Returns:
            numpy array of shape (batch_size, embedding_dim), or None if encoder not available
        """
        if not self.encoder_ready:
            logger.warning("Molecular encoder not available")
            return None

        if not smiles_list:
            return np.array([]).reshape(0, self.embedding_dim)

        try:
            return self.molecular_encoder.encode_batch(smiles_list)
        except Exception as e:
            logger.error(f"Failed to encode SMILES batch: {e}")
            return None

    async def get_similarity(self, smiles1: str, smiles2: str) -> Optional[float]:
        """
        Compute similarity between two molecules based on their embeddings.

        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string

        Returns:
            Cosine similarity score, or None if encoder not available
        """
        if not self.encoder_ready:
            return None

        try:
            return self.molecular_encoder.similarity(smiles1, smiles2)
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return None
