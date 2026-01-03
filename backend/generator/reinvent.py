"""
REINVENT-style Reinforcement Learning for Molecular Generation

Based on: Olivecrona et al. "Molecular De Novo Design through Deep Reinforcement Learning"
and REINVENT 2.0, 3.0, 4.0 improvements.
"""

import logging
import asyncio
import time
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import random
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ReinventConfig:
    """Configuration for REINVENT training"""
    # Learning
    learning_rate: float = 1e-4
    batch_size: int = 64
    sigma: float = 60.0              # Score threshold/baseline

    # Augmented likelihood
    prior_weight: float = 0.5        # Weight of prior likelihood in loss
    agent_weight: float = 1.0        # Weight of agent likelihood

    # Experience replay
    replay_buffer_size: int = 1000
    replay_batch_fraction: float = 0.25  # Fraction of batch from replay
    min_replay_score: float = 0.6    # Minimum score to add to replay

    # Diversity
    diversity_filter: bool = True
    similarity_threshold: float = 0.7  # Tanimoto threshold for diversity
    max_similar_in_batch: int = 5

    # Training control
    max_steps: int = 1000
    early_stop_patience: int = 50    # Stop if no improvement
    early_stop_threshold: float = 0.01

    # Regularization
    entropy_weight: float = 0.01     # Encourage exploration
    kl_weight: float = 0.1           # Limit divergence from prior
    max_kl: float = 0.5              # Maximum KL divergence allowed


@dataclass
class ReplayEntry:
    """Entry in experience replay buffer"""
    smiles: str
    score: float
    log_likelihood: float
    step: int


@dataclass
class TrainingMetrics:
    """Metrics from a training step"""
    step: int
    loss: float
    mean_score: float
    max_score: float
    valid_ratio: float
    unique_ratio: float
    diversity: float
    kl_divergence: float
    molecules_generated: int
    best_molecules: List[Dict]


class ExperienceReplay:
    """Experience replay buffer for high-scoring molecules"""

    def __init__(self, max_size: int = 1000, min_score: float = 0.6):
        self.buffer: deque = deque(maxlen=max_size)
        self.min_score = min_score
        self.smiles_set: set = set()  # For deduplication

    def add(self, smiles: str, score: float, log_likelihood: float, step: int):
        """Add a molecule to replay buffer if it meets criteria"""
        if score < self.min_score:
            return False

        if smiles in self.smiles_set:
            return False

        entry = ReplayEntry(smiles, score, log_likelihood, step)
        self.buffer.append(entry)
        self.smiles_set.add(smiles)

        # Maintain smiles_set size
        if len(self.smiles_set) > len(self.buffer):
            self.smiles_set = {e.smiles for e in self.buffer}

        return True

    def sample(self, n: int) -> List[ReplayEntry]:
        """Sample n entries, weighted by score"""
        if len(self.buffer) == 0:
            return []

        n = min(n, len(self.buffer))

        # Score-weighted sampling
        scores = np.array([e.score for e in self.buffer])
        probs = scores / scores.sum()

        indices = np.random.choice(len(self.buffer), size=n, replace=False, p=probs)
        return [self.buffer[i] for i in indices]

    def get_top(self, n: int) -> List[ReplayEntry]:
        """Get top n entries by score"""
        sorted_buffer = sorted(self.buffer, key=lambda x: x.score, reverse=True)
        return sorted_buffer[:n]

    def __len__(self) -> int:
        return len(self.buffer)


class DiversityFilter:
    """Filter to maintain molecular diversity during generation"""

    def __init__(self, similarity_threshold: float = 0.7, max_similar: int = 5):
        self.similarity_threshold = similarity_threshold
        self.max_similar = max_similar
        self.fingerprints: Dict[str, np.ndarray] = {}

    def _get_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """Get Morgan fingerprint for molecule"""
        if smiles in self.fingerprints:
            return self.fingerprints[smiles]

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_array = np.array(fp)
            self.fingerprints[smiles] = fp_array
            return fp_array

        except Exception:
            return None

    def _tanimoto(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Tanimoto similarity"""
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        return intersection / union if union > 0 else 0.0

    def filter_batch(self, smiles_list: List[str], scores: List[float]) -> Tuple[List[str], List[float], List[int]]:
        """
        Filter batch to maintain diversity.
        Returns filtered smiles, scores, and original indices.
        """
        if not smiles_list:
            return [], [], []

        # Sort by score (descending) to keep best molecules
        sorted_pairs = sorted(zip(smiles_list, scores, range(len(smiles_list))),
                             key=lambda x: x[1], reverse=True)

        filtered_smiles = []
        filtered_scores = []
        filtered_indices = []

        for smiles, score, idx in sorted_pairs:
            fp = self._get_fingerprint(smiles)
            if fp is None:
                continue

            # Check similarity to already selected molecules
            similar_count = 0
            for existing_smiles in filtered_smiles:
                existing_fp = self._get_fingerprint(existing_smiles)
                if existing_fp is not None:
                    sim = self._tanimoto(fp, existing_fp)
                    if sim > self.similarity_threshold:
                        similar_count += 1

            # Add if not too similar to existing selections
            if similar_count < self.max_similar:
                filtered_smiles.append(smiles)
                filtered_scores.append(score)
                filtered_indices.append(idx)

        return filtered_smiles, filtered_scores, filtered_indices

    def calculate_diversity(self, smiles_list: List[str]) -> float:
        """Calculate average pairwise diversity (1 - similarity)"""
        if len(smiles_list) < 2:
            return 1.0

        fps = [self._get_fingerprint(s) for s in smiles_list]
        fps = [fp for fp in fps if fp is not None]

        if len(fps) < 2:
            return 1.0

        similarities = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                similarities.append(self._tanimoto(fps[i], fps[j]))

        return 1.0 - np.mean(similarities) if similarities else 1.0

    def clear(self):
        """Clear fingerprint cache"""
        self.fingerprints.clear()


class ReinventTrainer:
    """
    REINVENT-style RL trainer for molecular generation.
    Fine-tunes an agent model toward high-scoring molecules while
    maintaining similarity to a frozen prior.
    """

    def __init__(self,
                 prior_model,
                 agent_model,
                 tokenizer,
                 scoring_function: Callable,
                 config: Optional[ReinventConfig] = None,
                 device: str = None):
        """
        Args:
            prior_model: Frozen pretrained model (provides baseline likelihood)
            agent_model: Trainable model (learns to generate good molecules)
            tokenizer: SMILES tokenizer
            scoring_function: Async function that takes SMILES and returns score dict
            config: Training configuration
            device: Device for training
        """
        self.prior = prior_model
        self.agent = agent_model
        self.tokenizer = tokenizer
        self.scoring_fn = scoring_function
        self.config = config or ReinventConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Freeze prior
        for param in self.prior.parameters():
            param.requires_grad = False
        self.prior.eval()

        # Setup agent optimizer
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=self.config.learning_rate
        )

        # Experience replay
        self.replay_buffer = ExperienceReplay(
            max_size=self.config.replay_buffer_size,
            min_score=self.config.min_replay_score
        )

        # Diversity filter
        self.diversity_filter = DiversityFilter(
            similarity_threshold=self.config.similarity_threshold,
            max_similar=self.config.max_similar_in_batch
        ) if self.config.diversity_filter else None

        # Training state
        self.current_step = 0
        self.best_score = 0.0
        self.steps_without_improvement = 0
        self.training_history: List[TrainingMetrics] = []
        self._stop_requested = False

    async def train_step(self) -> TrainingMetrics:
        """Execute one training step"""
        self.agent.train()

        # 1. Generate molecules from agent
        batch_size = self.config.batch_size
        replay_size = int(batch_size * self.config.replay_batch_fraction)
        gen_size = batch_size - replay_size

        # Generate new molecules
        generated_smiles, agent_log_probs = self._generate_batch(gen_size)

        # Add replay molecules
        replay_entries = self.replay_buffer.sample(replay_size)
        replay_smiles = [e.smiles for e in replay_entries]

        all_smiles = generated_smiles + replay_smiles

        # 2. Score molecules
        scores = await self._score_batch(all_smiles)

        # 3. Filter for validity and diversity
        valid_mask = [s > 0 for s in scores]
        valid_smiles = [sm for sm, v in zip(all_smiles, valid_mask) if v]
        valid_scores = [sc for sc, v in zip(scores, valid_mask) if v]

        if self.diversity_filter and valid_smiles:
            valid_smiles, valid_scores, _ = self.diversity_filter.filter_batch(
                valid_smiles, valid_scores
            )

        if not valid_smiles:
            return TrainingMetrics(
                step=self.current_step,
                loss=0.0,
                mean_score=0.0,
                max_score=0.0,
                valid_ratio=0.0,
                unique_ratio=0.0,
                diversity=0.0,
                kl_divergence=0.0,
                molecules_generated=len(generated_smiles),
                best_molecules=[]
            )

        # 4. Calculate likelihoods
        agent_nlls = self._calculate_nll(valid_smiles, self.agent)
        prior_nlls = self._calculate_nll(valid_smiles, self.prior)

        # 5. Calculate REINVENT loss
        scores_tensor = torch.tensor(valid_scores, device=self.device)

        # Augmented likelihood loss
        # Loss = -[S(x) - σ] × [log A(x) - log P(x)]
        score_diff = scores_tensor - self.config.sigma / 100.0  # Normalize sigma
        likelihood_diff = prior_nlls - agent_nlls  # Note: NLL so subtract

        loss = -torch.mean(score_diff * likelihood_diff)

        # KL regularization
        kl_div = torch.mean(agent_nlls - prior_nlls)
        if self.config.kl_weight > 0:
            loss += self.config.kl_weight * torch.clamp(kl_div, min=0, max=self.config.max_kl)

        # Entropy regularization (encourage exploration)
        if self.config.entropy_weight > 0:
            entropy = -torch.mean(agent_nlls)
            loss -= self.config.entropy_weight * entropy

        # 6. Update agent
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        self.optimizer.step()

        # 7. Update replay buffer
        for smiles, score in zip(valid_smiles[:gen_size], valid_scores[:gen_size]):
            agent_nll = self._calculate_nll([smiles], self.agent)[0].item()
            self.replay_buffer.add(smiles, score, -agent_nll, self.current_step)

        # 8. Calculate metrics
        diversity = self.diversity_filter.calculate_diversity(valid_smiles) if self.diversity_filter else 0.0
        unique_ratio = len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else 0.0

        # Get best molecules
        sorted_pairs = sorted(zip(valid_smiles, valid_scores), key=lambda x: x[1], reverse=True)
        best_molecules = [{"smiles": s, "score": sc} for s, sc in sorted_pairs[:5]]

        metrics = TrainingMetrics(
            step=self.current_step,
            loss=loss.item(),
            mean_score=np.mean(valid_scores),
            max_score=max(valid_scores),
            valid_ratio=len(valid_smiles) / len(all_smiles) if all_smiles else 0.0,
            unique_ratio=unique_ratio,
            diversity=diversity,
            kl_divergence=kl_div.item(),
            molecules_generated=len(generated_smiles),
            best_molecules=best_molecules
        )

        # Track improvement
        if metrics.max_score > self.best_score + self.config.early_stop_threshold:
            self.best_score = metrics.max_score
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        self.current_step += 1
        self.training_history.append(metrics)

        return metrics

    def _generate_batch(self, batch_size: int) -> Tuple[List[str], Optional[torch.Tensor]]:
        """Generate batch of SMILES from agent"""
        self.agent.eval()

        with torch.no_grad():
            # Use the agent's generate method
            if hasattr(self.agent, 'generate'):
                try:
                    # Try calling with our expected signature
                    outputs = self.agent.generate(
                        input_ids=torch.full((batch_size, 1),
                                           getattr(self.tokenizer, 'bos_token_id', 1),
                                           device=self.device),
                        max_length=128,
                        do_sample=True,
                        temperature=1.0,
                        pad_token_id=getattr(self.tokenizer, 'pad_token_id', 0),
                        eos_token_id=getattr(self.tokenizer, 'eos_token_id', 2)
                    )
                    sequences = outputs
                    log_probs = None
                except Exception:
                    # Fallback: generate one at a time
                    sequences = []
                    for _ in range(batch_size):
                        try:
                            output = self.agent.generate(
                                input_ids=torch.full((1, 1),
                                                   getattr(self.tokenizer, 'bos_token_id', 1),
                                                   device=self.device),
                                max_length=128,
                                do_sample=True,
                                temperature=1.0,
                                pad_token_id=getattr(self.tokenizer, 'pad_token_id', 0),
                                eos_token_id=getattr(self.tokenizer, 'eos_token_id', 2)
                            )
                            sequences.append(output[0])
                        except Exception:
                            pass
                    if sequences:
                        # Pad sequences to same length
                        max_len = max(len(s) for s in sequences)
                        padded = torch.full((len(sequences), max_len),
                                          getattr(self.tokenizer, 'pad_token_id', 0),
                                          device=self.device)
                        for i, seq in enumerate(sequences):
                            padded[i, :len(seq)] = seq
                        sequences = padded
                    else:
                        sequences = torch.zeros((0, 1), device=self.device, dtype=torch.long)
                    log_probs = None
            else:
                # Direct forward pass for custom models
                sequences = torch.zeros((batch_size, 1), device=self.device, dtype=torch.long)
                log_probs = None

        # Decode sequences
        smiles_list = []
        for seq in sequences:
            try:
                smiles = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
                if smiles:
                    smiles_list.append(smiles)
            except Exception:
                pass

        self.agent.train()
        return smiles_list, log_probs

    def _calculate_nll(self, smiles_list: List[str], model) -> torch.Tensor:
        """Calculate negative log-likelihood for SMILES under model"""
        model.eval()

        # Encode SMILES
        try:
            encoded = []
            for s in smiles_list:
                try:
                    enc = self.tokenizer.encode(s, add_special_tokens=True)
                    if isinstance(enc, torch.Tensor):
                        enc = enc.tolist()
                    encoded.append(enc)
                except Exception:
                    encoded.append([getattr(self.tokenizer, 'bos_token_id', 1)])

            if not encoded:
                return torch.zeros(0, device=self.device)

            max_len = max(len(e) for e in encoded)
            max_len = min(max_len, 128)  # Cap sequence length

            # Pad sequences
            pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
            padded = torch.full((len(encoded), max_len), pad_id, device=self.device)
            for i, e in enumerate(encoded):
                seq_len = min(len(e), max_len)
                padded[i, :seq_len] = torch.tensor(e[:seq_len], device=self.device)

            with torch.no_grad():
                # Get logits from model
                if hasattr(model, 'forward'):
                    outputs = model(padded)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                else:
                    logits = model(padded)

            # Calculate NLL
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            targets = padded[:, 1:]

            # Gather log probs for actual tokens
            token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

            # Mask padding
            mask = (targets != pad_id).float()

            # Sum NLL per sequence
            nll = -torch.sum(token_log_probs * mask, dim=1) / torch.sum(mask, dim=1).clamp(min=1)

            return nll

        except Exception as e:
            logger.warning(f"NLL calculation failed: {e}")
            return torch.zeros(len(smiles_list), device=self.device)

    async def _score_batch(self, smiles_list: List[str]) -> List[float]:
        """Score batch of SMILES using scoring function"""
        scores = []
        for smiles in smiles_list:
            try:
                result = await self.scoring_fn(smiles)
                # Calculate combined score and normalize to 0-1 range
                efficacy = result.get("efficacy", 0)
                safety = result.get("safety", 0)
                environmental = result.get("environmental", 0)
                sa_score = result.get("sa_score", 0)
                combined = (efficacy + safety + environmental + sa_score) / 4
                score = combined / 100.0
                scores.append(score)
            except Exception:
                scores.append(0.0)
        return scores

    async def train(self,
                    num_steps: Optional[int] = None,
                    callback: Optional[Callable] = None) -> List[TrainingMetrics]:
        """
        Run REINVENT training loop.

        Args:
            num_steps: Number of training steps (uses config if None)
            callback: Optional callback(metrics) called after each step

        Returns:
            List of training metrics
        """
        num_steps = num_steps or self.config.max_steps
        self._stop_requested = False

        logger.info(f"Starting REINVENT training for {num_steps} steps")

        for step in range(num_steps):
            if self._stop_requested:
                logger.info("Training stopped by request")
                break

            # Check early stopping
            if self.steps_without_improvement >= self.config.early_stop_patience:
                logger.info(f"Early stopping: no improvement for {self.config.early_stop_patience} steps")
                break

            # Train step
            metrics = await self.train_step()

            # Log progress
            if step % 10 == 0:
                logger.info(
                    f"Step {step}: loss={metrics.loss:.4f}, "
                    f"mean_score={metrics.mean_score:.3f}, "
                    f"max_score={metrics.max_score:.3f}, "
                    f"diversity={metrics.diversity:.3f}"
                )

            # Callback
            if callback:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics)
                else:
                    callback(metrics)

            # Small delay to prevent blocking
            await asyncio.sleep(0.01)

        logger.info(f"Training complete. Best score: {self.best_score:.3f}")
        return self.training_history

    def stop(self):
        """Request training stop"""
        self._stop_requested = True

    def get_best_molecules(self, n: int = 10) -> List[Dict]:
        """Get top n molecules from replay buffer"""
        top_entries = self.replay_buffer.get_top(n)
        return [
            {
                "smiles": e.smiles,
                "score": e.score,
                "step": e.step
            }
            for e in top_entries
        ]

    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "agent_state": self.agent.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": self.current_step,
            "best_score": self.best_score,
            "config": self.config
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.current_step = checkpoint["step"]
        self.best_score = checkpoint["best_score"]
        logger.info(f"Checkpoint loaded from {path}")
