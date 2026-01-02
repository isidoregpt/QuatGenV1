"""Pretrained Molecule Generator using HuggingFace models"""

import logging
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PretrainedMoleculeGenerator:
    """
    Wrapper for pretrained molecular generation models from HuggingFace.

    Uses the Franso/reinvent_171M_prior model by default, which is a
    transformer-based model trained on SMILES strings for molecule generation.
    """

    DEFAULT_MODEL = "Franso/reinvent_171M_prior"

    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the pretrained molecule generator.

        Args:
            model_name: HuggingFace model name/path. Defaults to Franso/reinvent_171M_prior
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._device = device
        self.model = None
        self.tokenizer = None
        self._is_ready = False
        self._fallback_mode = False

    @property
    def device(self) -> str:
        """Get the device being used."""
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def is_ready(self) -> bool:
        """Return True if model is loaded and ready for generation."""
        return self._is_ready

    @property
    def is_fallback(self) -> bool:
        """Return True if using fallback mode (model failed to load)."""
        return self._fallback_mode

    async def initialize(self) -> bool:
        """
        Async initialization for compatibility with existing engine.
        Downloads and loads the pretrained model from HuggingFace.

        Returns:
            True if model loaded successfully, False if fallback mode activated
        """
        try:
            logger.info(f"Loading pretrained model: {self.model_name}")
            logger.info(f"Target device: {self.device}")

            # Import transformers here to avoid import errors if not installed
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load tokenizer
            logger.info("Downloading/loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Ensure special tokens are set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.bos_token is None:
                self.tokenizer.bos_token = self.tokenizer.eos_token

            # Load model
            logger.info("Downloading/loading model weights...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Use float32 for stability
            )

            # Move to device
            try:
                self.model.to(self.device)
                logger.info(f"Model loaded on {self.device}")
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    logger.warning(f"CUDA error, falling back to CPU: {e}")
                    self._device = "cpu"
                    self.model.to("cpu")
                else:
                    raise

            self.model.eval()
            self._is_ready = True
            logger.info(f"Pretrained model ready: {self.model_name}")
            logger.info(f"Vocab size: {len(self.tokenizer)}")
            return True

        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            logger.warning("Falling back to random policy mode")
            self._fallback_mode = True
            self._is_ready = False
            return False

    def generate(
        self,
        batch_size: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        max_length: int = 128
    ) -> List[str]:
        """
        Generate SMILES strings using the pretrained model.

        Args:
            batch_size: Number of molecules to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            max_length: Maximum sequence length

        Returns:
            List of generated SMILES strings
        """
        if not self._is_ready:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        if self._fallback_mode:
            raise RuntimeError("Model in fallback mode. Use MoleculePolicy instead.")

        self.model.eval()

        with torch.no_grad():
            # Prepare input - start with BOS token or empty
            if self.tokenizer.bos_token_id is not None:
                input_ids = torch.tensor([[self.tokenizer.bos_token_id]] * batch_size,
                                        device=self.device)
            else:
                # Some models may not have BOS, start with pad or empty
                input_ids = torch.tensor([[self.tokenizer.pad_token_id]] * batch_size,
                                        device=self.device)

            # Generate sequences
            try:
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM during generation, reducing batch size")
                    # Try with smaller batch
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    outputs = self._generate_in_chunks(
                        batch_size, max_length, temperature, top_k, top_p
                    )
                else:
                    raise

            # Decode generated sequences
            smiles_list = []
            for seq in outputs:
                smiles = self.tokenizer.decode(seq, skip_special_tokens=True)
                # Clean up SMILES string
                smiles = smiles.strip()
                smiles_list.append(smiles)

            return smiles_list

    def _generate_in_chunks(
        self,
        batch_size: int,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        chunk_size: int = 8
    ) -> torch.Tensor:
        """Generate in smaller chunks to avoid OOM errors."""
        all_outputs = []

        for i in range(0, batch_size, chunk_size):
            current_batch = min(chunk_size, batch_size - i)

            if self.tokenizer.bos_token_id is not None:
                input_ids = torch.tensor(
                    [[self.tokenizer.bos_token_id]] * current_batch,
                    device=self.device
                )
            else:
                input_ids = torch.tensor(
                    [[self.tokenizer.pad_token_id]] * current_batch,
                    device=self.device
                )

            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            all_outputs.append(outputs)

        # Pad and concatenate
        max_len = max(o.shape[1] for o in all_outputs)
        padded = []
        for o in all_outputs:
            if o.shape[1] < max_len:
                padding = torch.full(
                    (o.shape[0], max_len - o.shape[1]),
                    self.tokenizer.pad_token_id,
                    device=self.device
                )
                o = torch.cat([o, padding], dim=1)
            padded.append(o)

        return torch.cat(padded, dim=0)

    def generate_from_scaffold(
        self,
        scaffold_smiles: str,
        num_samples: int = 10,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        max_length: int = 128
    ) -> List[str]:
        """
        Generate molecules starting from a scaffold SMILES.

        This method uses the scaffold as a prompt and generates completions.

        Args:
            scaffold_smiles: Starting SMILES scaffold
            num_samples: Number of molecules to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            max_length: Maximum sequence length

        Returns:
            List of generated SMILES strings
        """
        if not self._is_ready:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        if self._fallback_mode:
            raise RuntimeError("Model in fallback mode. Use MoleculePolicy instead.")

        self.model.eval()

        with torch.no_grad():
            # Tokenize scaffold
            scaffold_tokens = self.tokenizer.encode(
                scaffold_smiles,
                add_special_tokens=True,
                return_tensors="pt"
            )

            # Expand to batch size
            input_ids = scaffold_tokens.repeat(num_samples, 1).to(self.device)

            # Generate completions
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Decode
            smiles_list = []
            for seq in outputs:
                smiles = self.tokenizer.decode(seq, skip_special_tokens=True)
                smiles = smiles.strip()
                smiles_list.append(smiles)

            return smiles_list

    def generate_with_log_probs(
        self,
        batch_size: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        max_length: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences and return both tokens and log probabilities.

        This is needed for RL training where we need the log probs of actions.

        Args:
            batch_size: Number of molecules to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            max_length: Maximum sequence length

        Returns:
            Tuple of (sequences tensor, log_probs tensor)
        """
        if not self._is_ready:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        if self._fallback_mode:
            raise RuntimeError("Model in fallback mode. Use MoleculePolicy instead.")

        self.model.eval()

        # Start with BOS tokens
        if self.tokenizer.bos_token_id is not None:
            sequences = torch.full(
                (batch_size, 1),
                self.tokenizer.bos_token_id,
                dtype=torch.long,
                device=self.device
            )
        else:
            sequences = torch.full(
                (batch_size, 1),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device
            )

        log_probs_list = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            for _ in range(max_length - 1):
                # Get logits from model
                outputs = self.model(sequences)
                logits = outputs.logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Sample from distribution
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

                # Get log probs of selected tokens
                log_probs = torch.log(probs.gather(1, next_tokens) + 1e-10)
                log_probs_list.append(log_probs)

                # Append to sequences
                sequences = torch.cat([sequences, next_tokens], dim=1)

                # Check for EOS
                if self.tokenizer.eos_token_id is not None:
                    finished = finished | (next_tokens.squeeze(-1) == self.tokenizer.eos_token_id)
                    if finished.all():
                        break

        # Stack log probs
        log_probs_tensor = torch.cat(log_probs_list, dim=1)

        return sequences, log_probs_tensor

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.tokenizer is None:
            return 0
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        """Get pad token ID."""
        if self.tokenizer is None:
            return 0
        return self.tokenizer.pad_token_id or 0

    @property
    def bos_token_id(self) -> int:
        """Get beginning of sequence token ID."""
        if self.tokenizer is None:
            return 0
        return self.tokenizer.bos_token_id or self.tokenizer.pad_token_id or 0

    @property
    def eos_token_id(self) -> int:
        """Get end of sequence token ID."""
        if self.tokenizer is None:
            return 0
        return self.tokenizer.eos_token_id or 0

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to SMILES string."""
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def encode(self, smiles: str, max_length: int = None) -> List[int]:
        """Encode SMILES string to token IDs."""
        if self.tokenizer is None:
            return []

        encoded = self.tokenizer.encode(
            smiles,
            add_special_tokens=True,
            max_length=max_length,
            truncation=max_length is not None
        )

        # Pad if necessary
        if max_length and len(encoded) < max_length:
            encoded = encoded + [self.pad_token_id] * (max_length - len(encoded))

        return encoded
