"""Embedding-based Property Predictors using MLP heads on molecular embeddings"""

import logging
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from scoring.molecular_encoder import MolecularEncoder

logger = logging.getLogger(__name__)


class EmbeddingPredictor(nn.Module):
    """
    MLP head for property prediction from molecular embeddings.

    Takes embeddings from MolecularEncoder and predicts a single property value.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize the embedding predictor.

        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of output (1 for regression, n for classification)
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', or 'tanh')
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Select activation function
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        else:
            act_fn = nn.ReLU()

        # Build MLP: input -> hidden -> output
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the predictor.

        Args:
            embeddings: Tensor of shape (batch_size, input_dim)

        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        return self.network(embeddings)


class PropertyPredictorHead:
    """
    Wrapper combining MolecularEncoder with multiple prediction heads.

    Provides a unified interface for predicting multiple molecular properties
    from SMILES strings using learned embeddings.
    """

    def __init__(
        self,
        encoder: MolecularEncoder,
        device: str = None
    ):
        """
        Initialize the property predictor.

        Args:
            encoder: Initialized MolecularEncoder instance
            device: Device to use for prediction heads
        """
        self.encoder = encoder
        self._device = device
        self.heads: Dict[str, EmbeddingPredictor] = {}
        self._is_ready = False

    @property
    def device(self) -> str:
        """Get the device being used."""
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def is_ready(self) -> bool:
        """Return True if encoder and heads are ready."""
        return self._is_ready and self.encoder.is_ready

    @property
    def available_properties(self) -> List[str]:
        """Return list of property names with trained heads."""
        return list(self.heads.keys())

    def add_head(
        self,
        property_name: str,
        output_dim: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ) -> EmbeddingPredictor:
        """
        Add a prediction head for a specific property.

        Args:
            property_name: Name of the property to predict
            output_dim: Output dimension (1 for regression)
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability

        Returns:
            The created EmbeddingPredictor
        """
        if not self.encoder.is_ready:
            raise RuntimeError("Encoder not initialized")

        head = EmbeddingPredictor(
            input_dim=self.encoder.embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )
        head.to(self.device)
        head.eval()

        self.heads[property_name] = head
        logger.info(f"Added prediction head for: {property_name}")

        self._is_ready = True
        return head

    def remove_head(self, property_name: str) -> bool:
        """
        Remove a prediction head.

        Args:
            property_name: Name of the property head to remove

        Returns:
            True if removed, False if not found
        """
        if property_name in self.heads:
            del self.heads[property_name]
            logger.info(f"Removed prediction head for: {property_name}")
            return True
        return False

    def predict(self, smiles: str, property_name: str) -> float:
        """
        Predict a property value for a single molecule.

        Args:
            smiles: SMILES string
            property_name: Name of the property to predict

        Returns:
            Predicted property value
        """
        if property_name not in self.heads:
            raise ValueError(f"No head for property: {property_name}")

        predictions = self.predict_batch([smiles], property_name)
        return float(predictions[0])

    def predict_batch(
        self,
        smiles_list: List[str],
        property_name: str
    ) -> np.ndarray:
        """
        Predict property values for a batch of molecules.

        Args:
            smiles_list: List of SMILES strings
            property_name: Name of the property to predict

        Returns:
            numpy array of predicted values
        """
        if property_name not in self.heads:
            raise ValueError(f"No head for property: {property_name}")

        if not smiles_list:
            return np.array([])

        head = self.heads[property_name]
        head.eval()

        # Get embeddings
        embeddings = self.encoder.encode_batch(smiles_list)
        embeddings_tensor = torch.tensor(
            embeddings,
            dtype=torch.float32,
            device=self.device
        )

        # Predict
        with torch.no_grad():
            predictions = head(embeddings_tensor)

        return predictions.cpu().numpy().squeeze(-1)

    def predict_all(self, smiles: str) -> Dict[str, float]:
        """
        Predict all available properties for a molecule.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary mapping property names to predicted values
        """
        if not self.heads:
            return {}

        # Get embedding once
        embedding = self.encoder.encode(smiles)
        embedding_tensor = torch.tensor(
            embedding,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        results = {}
        with torch.no_grad():
            for prop_name, head in self.heads.items():
                head.eval()
                pred = head(embedding_tensor)
                results[prop_name] = float(pred.cpu().numpy().squeeze())

        return results

    def predict_all_batch(
        self,
        smiles_list: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Predict all available properties for a batch of molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dictionary mapping property names to arrays of predicted values
        """
        if not self.heads or not smiles_list:
            return {}

        # Get embeddings once
        embeddings = self.encoder.encode_batch(smiles_list)
        embeddings_tensor = torch.tensor(
            embeddings,
            dtype=torch.float32,
            device=self.device
        )

        results = {}
        with torch.no_grad():
            for prop_name, head in self.heads.items():
                head.eval()
                preds = head(embeddings_tensor)
                results[prop_name] = preds.cpu().numpy().squeeze(-1)

        return results

    def train_head(
        self,
        property_name: str,
        smiles_list: List[str],
        targets: List[float],
        epochs: int = 100,
        learning_rate: float = 1e-3,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Train a prediction head on labeled data.

        Args:
            property_name: Name of the property to train
            smiles_list: Training SMILES strings
            targets: Target values for each molecule
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size

        Returns:
            Dictionary with training metrics
        """
        if property_name not in self.heads:
            self.add_head(property_name)

        head = self.heads[property_name]
        head.train()

        # Get all embeddings
        logger.info(f"Computing embeddings for {len(smiles_list)} molecules...")
        all_embeddings = self.encoder.encode_batch(smiles_list)
        all_embeddings = torch.tensor(
            all_embeddings,
            dtype=torch.float32,
            device=self.device
        )
        all_targets = torch.tensor(
            targets,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(-1)

        # Setup optimizer and loss
        optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        n_samples = len(smiles_list)
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle data
            indices = torch.randperm(n_samples)
            all_embeddings = all_embeddings[indices]
            all_targets = all_targets[indices]

            for i in range(0, n_samples, batch_size):
                batch_emb = all_embeddings[i:i + batch_size]
                batch_targets = all_targets[i:i + batch_size]

                optimizer.zero_grad()
                predictions = head(batch_emb)
                loss = criterion(predictions, batch_targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        head.eval()
        logger.info(f"Training complete for {property_name}")

        return {
            "final_loss": losses[-1],
            "epochs": epochs,
            "samples": n_samples
        }

    def save_head(self, property_name: str, path: str):
        """Save a prediction head to disk."""
        if property_name not in self.heads:
            raise ValueError(f"No head for property: {property_name}")

        torch.save(self.heads[property_name].state_dict(), path)
        logger.info(f"Saved head for {property_name} to {path}")

    def load_head(
        self,
        property_name: str,
        path: str,
        output_dim: int = 1,
        hidden_dim: int = 256
    ):
        """Load a prediction head from disk."""
        head = EmbeddingPredictor(
            input_dim=self.encoder.embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        head.load_state_dict(torch.load(path, map_location=self.device))
        head.to(self.device)
        head.eval()

        self.heads[property_name] = head
        self._is_ready = True
        logger.info(f"Loaded head for {property_name} from {path}")
