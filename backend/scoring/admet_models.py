"""ADMET Property Prediction using ChemFM models from HuggingFace"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ADMETModelConfig:
    """Configuration for a single ADMET prediction model"""
    model_name: str
    property_name: str
    task_type: str  # "classification" or "regression"
    output_transform: Optional[str] = None  # "sigmoid", "softmax", or None
    threshold: Optional[float] = None  # For classification
    description: str = ""


# Define all ADMET models to integrate
ADMET_MODELS: Dict[str, ADMETModelConfig] = {
    "ld50": ADMETModelConfig(
        model_name="ChemFM/admet_ld50_zhu",
        property_name="LD50",
        task_type="regression",
        description="Acute oral toxicity in rats (log mg/kg)"
    ),
    "herg": ADMETModelConfig(
        model_name="ChemFM/admet_herg",
        property_name="hERG_inhibition",
        task_type="classification",
        output_transform="sigmoid",
        threshold=0.5,
        description="hERG potassium channel inhibition (cardiotoxicity risk)"
    ),
    "ames": ADMETModelConfig(
        model_name="ChemFM/admet_ames",
        property_name="Ames_mutagenicity",
        task_type="classification",
        output_transform="sigmoid",
        threshold=0.5,
        description="Ames test mutagenicity prediction"
    ),
    "dili": ADMETModelConfig(
        model_name="ChemFM/admet_dili",
        property_name="DILI_risk",
        task_type="classification",
        output_transform="sigmoid",
        threshold=0.5,
        description="Drug-induced liver injury risk"
    ),
    "bbb": ADMETModelConfig(
        model_name="ChemFM/admet_bbb_martins",
        property_name="BBB_penetration",
        task_type="classification",
        output_transform="sigmoid",
        threshold=0.5,
        description="Blood-brain barrier penetration"
    ),
    "solubility": ADMETModelConfig(
        model_name="ChemFM/admet_solubility_aqsoldb",
        property_name="Solubility",
        task_type="regression",
        description="Aqueous solubility (log mol/L)"
    ),
    "lipophilicity": ADMETModelConfig(
        model_name="ChemFM/admet_lipophilicity_astrazeneca",
        property_name="Lipophilicity",
        task_type="regression",
        description="Lipophilicity (logD)"
    ),
    "clearance": ADMETModelConfig(
        model_name="ChemFM/admet_clearance_microsome_az",
        property_name="Clearance",
        task_type="regression",
        description="Microsomal metabolic clearance"
    ),
}


class ADMETPredictor:
    """
    Unified ADMET property predictor using ChemFM models from HuggingFace.

    Loads multiple ADMET prediction models and provides a unified interface
    for predicting various toxicity and pharmacokinetic properties.
    """

    def __init__(
        self,
        models_to_load: Optional[List[str]] = None,
        device: str = None,
        lazy_load: bool = False
    ):
        """
        Initialize the ADMET predictor.

        Args:
            models_to_load: List of model keys to load (e.g., ["ld50", "herg", "ames"])
                           If None, loads all available models
            device: Device to use ("cuda" or "cpu"). Auto-detects if None.
            lazy_load: If True, models are loaded on first use instead of initialization
        """
        self._device = device
        self.models_to_load = models_to_load or list(ADMET_MODELS.keys())
        self.lazy_load = lazy_load
        self.models: Dict[str, Tuple[Any, Any]] = {}  # {key: (model, tokenizer)}
        self.configs: Dict[str, ADMETModelConfig] = {}
        self._is_ready = False
        self._loading_errors: Dict[str, str] = {}

    @property
    def device(self) -> str:
        """Get the device being used."""
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def is_ready(self) -> bool:
        """Return True if at least one model is loaded."""
        return self._is_ready

    @property
    def available_properties(self) -> List[str]:
        """Return list of loaded model keys."""
        return list(self.models.keys())

    @property
    def loading_errors(self) -> Dict[str, str]:
        """Return dict of models that failed to load and their error messages."""
        return self._loading_errors

    async def initialize(self) -> bool:
        """
        Load all specified models from HuggingFace.

        Returns:
            True if at least one model loaded successfully
        """
        if self.lazy_load:
            # In lazy mode, just mark as ready - models load on first use
            self._is_ready = True
            logger.info(f"ADMET predictor initialized in lazy mode for {len(self.models_to_load)} models")
            return True

        logger.info(f"Loading ADMET models: {self.models_to_load}")
        logger.info(f"Target device: {self.device}")

        for key in self.models_to_load:
            await self._load_model(key)

        self._is_ready = len(self.models) > 0
        logger.info(f"ADMET predictor ready with {len(self.models)}/{len(self.models_to_load)} models")

        if self._loading_errors:
            logger.warning(f"Failed to load {len(self._loading_errors)} models: {list(self._loading_errors.keys())}")

        return self._is_ready

    async def _load_model(self, key: str) -> bool:
        """Load a single ADMET model."""
        if key in self.models:
            return True  # Already loaded

        if key not in ADMET_MODELS:
            logger.warning(f"Unknown ADMET model key: {key}")
            self._loading_errors[key] = "Unknown model key"
            return False

        config = ADMET_MODELS[key]

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Loading {config.model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                trust_remote_code=True
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )

            # Move to device
            try:
                model.to(self.device)
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    logger.warning(f"CUDA error for {key}, falling back to CPU: {e}")
                    self._device = "cpu"
                    model.to("cpu")
                else:
                    raise

            model.eval()

            self.models[key] = (model, tokenizer)
            self.configs[key] = config
            logger.info(f"Loaded {key}: {config.description}")
            return True

        except Exception as e:
            logger.error(f"Failed to load {config.model_name}: {e}")
            self._loading_errors[key] = str(e)
            return False

    def predict(self, smiles: str, property_key: str) -> Dict:
        """
        Predict a single ADMET property for a molecule.

        Args:
            smiles: SMILES string of the molecule
            property_key: Key of the property to predict (e.g., "ld50", "herg")

        Returns:
            Dictionary with prediction results
        """
        # Lazy load if needed
        if self.lazy_load and property_key not in self.models:
            if property_key in self.models_to_load:
                import asyncio
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(self._load_model(property_key))
                finally:
                    loop.close()

        if property_key not in self.models:
            error_msg = self._loading_errors.get(property_key, "Model not loaded")
            return {"error": error_msg, "property": property_key}

        model, tokenizer = self.models[property_key]
        config = self.configs[property_key]

        try:
            # Tokenize input
            inputs = tokenizer(
                smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            # Transform output based on task type
            if config.task_type == "classification":
                return self._process_classification_output(logits, config)
            else:
                return self._process_regression_output(logits, config)

        except Exception as e:
            logger.error(f"Prediction error for {property_key}: {e}")
            return {"error": str(e), "property": config.property_name}

    def _process_classification_output(
        self,
        logits: torch.Tensor,
        config: ADMETModelConfig
    ) -> Dict:
        """Process classification model output."""
        if config.output_transform == "sigmoid":
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        elif config.output_transform == "softmax":
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        else:
            probs = logits.cpu().numpy()[0]

        # Handle different output shapes
        if isinstance(probs, np.ndarray):
            if probs.ndim == 0:
                probability = float(probs)
            elif probs.shape[0] == 1:
                probability = float(probs[0])
            elif probs.shape[0] == 2:
                # Binary classification: use probability of positive class
                probability = float(probs[1])
            else:
                probability = float(probs.max())
        else:
            probability = float(probs)

        threshold = config.threshold or 0.5
        prediction = probability > threshold

        return {
            "property": config.property_name,
            "prediction": bool(prediction),
            "probability": round(probability, 4),
            "threshold": threshold,
            "task_type": "classification",
            "description": config.description
        }

    def _process_regression_output(
        self,
        logits: torch.Tensor,
        config: ADMETModelConfig
    ) -> Dict:
        """Process regression model output."""
        output = logits.cpu().numpy()

        # Handle different output shapes
        if output.ndim == 2:
            value = float(output[0][0])
        elif output.ndim == 1:
            value = float(output[0])
        else:
            value = float(output)

        return {
            "property": config.property_name,
            "value": round(value, 4),
            "task_type": "regression",
            "description": config.description
        }

    def predict_all(self, smiles: str) -> Dict[str, Dict]:
        """
        Predict all loaded ADMET properties for a molecule.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Dictionary mapping property keys to prediction results
        """
        results = {}

        # In lazy mode, try all requested models
        keys_to_predict = self.models_to_load if self.lazy_load else list(self.models.keys())

        for key in keys_to_predict:
            results[key] = self.predict(smiles, key)

        return results

    def predict_batch(
        self,
        smiles_list: List[str],
        property_key: str
    ) -> List[Dict]:
        """
        Batch prediction for a single property.

        Args:
            smiles_list: List of SMILES strings
            property_key: Key of the property to predict

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(s, property_key) for s in smiles_list]

    def predict_all_batch(
        self,
        smiles_list: List[str]
    ) -> List[Dict[str, Dict]]:
        """
        Predict all properties for a batch of molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of dictionaries, each mapping property keys to predictions
        """
        return [self.predict_all(s) for s in smiles_list]

    def get_model_info(self) -> Dict[str, Dict]:
        """Get information about all configured models."""
        info = {}
        for key, config in ADMET_MODELS.items():
            info[key] = {
                "model_name": config.model_name,
                "property_name": config.property_name,
                "task_type": config.task_type,
                "description": config.description,
                "loaded": key in self.models,
                "error": self._loading_errors.get(key)
            }
        return info
