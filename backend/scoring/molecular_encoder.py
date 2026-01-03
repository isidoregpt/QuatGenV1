"""Molecular Encoder using ChemBERTa for SMILES embedding generation"""

import logging
from typing import List, Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)


class MolecularEncoder:
    """
    Molecular encoder using ChemBERTa to convert SMILES strings to embeddings.

    Uses DeepChem/ChemBERTa-77M-MLM by default, which provides rich molecular
    representations trained on millions of molecules.
    """

    DEFAULT_MODEL = "DeepChem/ChemBERTa-77M-MLM"

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        pooling: str = "mean",
        max_length: int = 512
    ):
        """
        Initialize the molecular encoder.

        Args:
            model_name: HuggingFace model name/path. Defaults to ChemBERTa-77M-MLM
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            pooling: Pooling strategy - 'mean' or 'cls'
            max_length: Maximum token sequence length
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._device = device
        self.pooling = pooling
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self._is_ready = False
        self._embedding_dim = 0

    @property
    def device(self) -> str:
        """Get the device being used."""
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def is_ready(self) -> bool:
        """Return True if model is loaded and ready for encoding."""
        return self._is_ready

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        return self._embedding_dim

    async def initialize(self) -> bool:
        """
        Async initialization for compatibility with existing pipeline.
        Downloads and loads the ChemBERTa model from HuggingFace.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading molecular encoder: {self.model_name}")
            logger.info(f"Target device: {self.device}")
            logger.info(f"Pooling strategy: {self.pooling}")

            # Import transformers here to avoid import errors if not installed
            from transformers import AutoModel, AutoTokenizer

            # Load tokenizer
            logger.info("Downloading/loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model (use AutoModel for encoder, not CausalLM)
            logger.info("Downloading/loading model weights...")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )

            # Move to device
            try:
                self.model.to(self.device)
                logger.info(f"Molecular encoder loaded on {self.device}")
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    logger.warning(f"CUDA error, falling back to CPU: {e}")
                    self._device = "cpu"
                    self.model.to("cpu")
                else:
                    raise

            self.model.eval()

            # Get embedding dimension from model config
            self._embedding_dim = self.model.config.hidden_size
            logger.info(f"Embedding dimension: {self._embedding_dim}")

            self._is_ready = True
            logger.info(f"Molecular encoder ready: {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load molecular encoder: {e}")
            logger.warning("Molecular encoding will be unavailable")
            self._is_ready = False
            return False

    def encode(self, smiles: str) -> np.ndarray:
        """
        Convert a single SMILES string to an embedding vector.

        Args:
            smiles: SMILES string to encode

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not self._is_ready:
            raise RuntimeError("Encoder not initialized. Call initialize() first.")

        embeddings = self.encode_batch([smiles])
        return embeddings[0]

    def encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Convert a batch of SMILES strings to embeddings.

        Args:
            smiles_list: List of SMILES strings to encode

        Returns:
            numpy array of shape (batch_size, embedding_dim)
        """
        if not self._is_ready:
            raise RuntimeError("Encoder not initialized. Call initialize() first.")

        if not smiles_list:
            return np.array([]).reshape(0, self._embedding_dim)

        self.model.eval()

        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                smiles_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )

            # Log warning if truncation occurred
            for i, smiles in enumerate(smiles_list):
                tokens = self.tokenizer.encode(smiles, add_special_tokens=True)
                if len(tokens) > self.max_length:
                    logger.warning(
                        f"SMILES truncated from {len(tokens)} to {self.max_length} tokens: {smiles[:50]}..."
                    )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            try:
                # Get model outputs
                outputs = self.model(**inputs)

                # Extract embeddings based on pooling strategy
                if self.pooling == "cls":
                    # Use [CLS] token embedding (first token)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                else:
                    # Mean pooling over all tokens (excluding padding)
                    embeddings = self._mean_pooling(
                        outputs.last_hidden_state,
                        inputs['attention_mask']
                    )

                return embeddings.cpu().numpy()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM during encoding, processing in smaller batches")
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    return self._encode_in_chunks(smiles_list)
                else:
                    raise

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform mean pooling over token embeddings, accounting for padding.

        Args:
            token_embeddings: Tensor of shape (batch, seq_len, hidden_dim)
            attention_mask: Tensor of shape (batch, seq_len)

        Returns:
            Tensor of shape (batch, hidden_dim)
        """
        # Expand attention mask to match embedding dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()

        # Sum embeddings weighted by attention mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

        # Clamp to avoid division by zero
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask

    def _encode_in_chunks(
        self,
        smiles_list: List[str],
        chunk_size: int = 8
    ) -> np.ndarray:
        """Encode in smaller chunks to avoid OOM errors."""
        all_embeddings = []

        for i in range(0, len(smiles_list), chunk_size):
            chunk = smiles_list[i:i + chunk_size]

            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)

            if self.pooling == "cls":
                embeddings = outputs.last_hidden_state[:, 0, :]
            else:
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state,
                    inputs['attention_mask']
                )

            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def get_attention_weights(self, smiles: str) -> Optional[np.ndarray]:
        """
        Get attention weights for a SMILES string for interpretability.

        Args:
            smiles: SMILES string to analyze

        Returns:
            numpy array of attention weights, or None if not available
        """
        if not self._is_ready:
            raise RuntimeError("Encoder not initialized. Call initialize() first.")

        self.model.eval()

        with torch.no_grad():
            inputs = self.tokenizer(
                smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get outputs with attention
            outputs = self.model(**inputs, output_attentions=True)

            if outputs.attentions is None:
                return None

            # Average attention weights across all layers and heads
            # Shape: (num_layers, batch, num_heads, seq_len, seq_len)
            attention_weights = torch.stack(outputs.attentions)

            # Average across layers and heads
            avg_attention = attention_weights.mean(dim=(0, 2))  # (batch, seq_len, seq_len)

            # Get attention from [CLS] token to other tokens
            cls_attention = avg_attention[0, 0, :]  # (seq_len,)

            return cls_attention.cpu().numpy()

    def get_token_embeddings(self, smiles: str) -> Tuple[List[str], np.ndarray]:
        """
        Get individual token embeddings for a SMILES string.

        Args:
            smiles: SMILES string to encode

        Returns:
            Tuple of (list of tokens, embeddings array of shape (num_tokens, embedding_dim))
        """
        if not self._is_ready:
            raise RuntimeError("Encoder not initialized. Call initialize() first.")

        self.model.eval()

        with torch.no_grad():
            inputs = self.tokenizer(
                smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)

            # Get token embeddings
            token_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)

            # Get tokens
            token_ids = inputs['input_ids'][0].cpu().tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

            # Remove padding tokens
            attention_mask = inputs['attention_mask'][0].cpu().tolist()
            valid_length = sum(attention_mask)

            tokens = tokens[:valid_length]
            embeddings = token_embeddings[:valid_length].cpu().numpy()

            return tokens, embeddings

    def similarity(self, smiles1: str, smiles2: str) -> float:
        """
        Compute cosine similarity between two molecules based on their embeddings.

        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string

        Returns:
            Cosine similarity score between -1 and 1
        """
        emb1 = self.encode(smiles1)
        emb2 = self.encode(smiles2)

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def batch_similarity(
        self,
        query_smiles: str,
        smiles_list: List[str]
    ) -> np.ndarray:
        """
        Compute similarity between a query molecule and a list of molecules.

        Args:
            query_smiles: Query SMILES string
            smiles_list: List of SMILES strings to compare against

        Returns:
            numpy array of similarity scores
        """
        if not smiles_list:
            return np.array([])

        query_emb = self.encode(query_smiles)
        target_embs = self.encode_batch(smiles_list)

        # Normalize embeddings
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        target_norms = target_embs / (np.linalg.norm(target_embs, axis=1, keepdims=True) + 1e-9)

        # Compute cosine similarities
        similarities = np.dot(target_norms, query_norm)

        return similarities
