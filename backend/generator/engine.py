"""Generator Engine - Main RL-based molecule generation loop"""

import asyncio
import time
import logging
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field
import torch
import numpy as np

from generator.tokenizer import SMIAISTokenizer
from generator.policy import MoleculePolicy, PolicyOptimizer
from generator.constraints import QuatConstraints, validate_quat
from generator.pretrained_model import PretrainedMoleculeGenerator
from generator.reinvent import ReinventTrainer, ReinventConfig, TrainingMetrics
from generator.filters import MolecularFilter, FilterConfig, DiversitySelector
from scoring.pipeline import ScoringPipeline
from database.connection import get_db_context
from database import queries

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    # Pretrained model settings
    use_pretrained: bool = True
    pretrained_model_name: str = "Franso/reinvent_171M_prior"
    # Random policy settings (used as fallback or when use_pretrained=False)
    vocab_size: int = 500
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_len: int = 128
    # Generation settings
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    # Training settings
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_weight: float = 1.0
    value_weight: float = 0.5
    # REINVENT RL fine-tuning settings
    use_rl_finetuning: bool = True
    rl_sigma: float = 60.0
    rl_replay_buffer_size: int = 1000
    rl_diversity_filter: bool = True
    rl_similarity_threshold: float = 0.7
    rl_early_stop_patience: int = 50
    rl_kl_weight: float = 0.1
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 8


@dataclass
class GenerationState:
    is_running: bool = False
    molecules_generated: int = 0
    molecules_valid: int = 0
    current_batch: int = 0
    total_batches: int = 0
    start_time: float = 0.0
    pareto_frontier: List[Dict] = field(default_factory=list)
    best_efficacy: float = 0.0
    best_safety: float = 0.0
    best_environmental: float = 0.0
    best_combined: float = 0.0
    
    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0
    
    @property
    def molecules_per_hour(self) -> float:
        return self.molecules_generated / self.elapsed_seconds * 3600 if self.elapsed_seconds > 1 else 0.0
    
    @property
    def estimated_remaining_seconds(self) -> float:
        if self.molecules_per_hour < 1:
            return 0.0
        return (self.total_batches - self.current_batch) * 64 / self.molecules_per_hour * 3600


class GeneratorEngine:
    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self.state = GenerationState()
        self.tokenizer: Optional[SMIAISTokenizer] = None
        self.policy: Optional[MoleculePolicy] = None
        self.optimizer: Optional[PolicyOptimizer] = None
        self.scoring: Optional[ScoringPipeline] = None
        self.pretrained_generator: Optional[PretrainedMoleculeGenerator] = None
        self._use_pretrained_for_generation = False
        self._stop_requested = False
        # REINVENT RL fine-tuning
        self.reinvent_trainer: Optional[ReinventTrainer] = None
        self.rl_training_active = False
        self._rl_metrics: Optional[TrainingMetrics] = None
        # Molecular filtering and diversity
        self.molecular_filter: Optional[MolecularFilter] = None
        self.diversity_selector: Optional[DiversitySelector] = None
    
    @property
    def is_ready(self) -> bool:
        generator_ready = self.policy is not None or (
            self.pretrained_generator is not None and self.pretrained_generator.is_ready
        )
        return generator_ready and self.scoring is not None

    @property
    def using_pretrained(self) -> bool:
        return self._use_pretrained_for_generation
    
    @property
    def is_running(self) -> bool:
        return self.state.is_running
    
    @property
    def scoring_ready(self) -> bool:
        return self.scoring is not None and self.scoring.is_ready
    
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size if self.tokenizer else 0
    
    @property
    def device(self) -> str:
        return self.config.device
    
    @property
    def molecules_generated(self) -> int:
        return self.state.molecules_generated
    
    @property
    def molecules_per_hour(self) -> float:
        return self.state.molecules_per_hour
    
    @property
    def pareto_frontier_size(self) -> int:
        return len(self.state.pareto_frontier)
    
    @property
    def current_batch(self) -> int:
        return self.state.current_batch
    
    @property
    def total_batches(self) -> int:
        return self.state.total_batches
    
    @property
    def elapsed_seconds(self) -> float:
        return self.state.elapsed_seconds
    
    @property
    def estimated_remaining_seconds(self) -> float:
        return self.state.estimated_remaining_seconds
    
    @property
    def top_scores(self) -> dict:
        return {"efficacy": self.state.best_efficacy, "safety": self.state.best_safety,
                "environmental": self.state.best_environmental, "combined": self.state.best_combined}

    @property
    def rl_training_status(self) -> dict:
        """Get current RL training status and metrics"""
        if not self.reinvent_trainer:
            return {"available": False, "active": False}

        status = {
            "available": True,
            "active": self.rl_training_active,
            "config": {
                "sigma": self.config.rl_sigma,
                "learning_rate": self.config.learning_rate,
                "diversity_filter": self.config.rl_diversity_filter,
                "similarity_threshold": self.config.rl_similarity_threshold,
            }
        }

        if self._rl_metrics:
            status["metrics"] = {
                "step": self._rl_metrics.step,
                "mean_score": self._rl_metrics.mean_score,
                "max_score": self._rl_metrics.max_score,
                "valid_ratio": self._rl_metrics.valid_ratio,
                "unique_ratio": self._rl_metrics.unique_ratio,
                "diversity": self._rl_metrics.diversity,
                "mean_nll": self._rl_metrics.mean_nll,
                "kl_divergence": self._rl_metrics.kl_divergence,
            }

        if self.reinvent_trainer:
            status["replay_buffer_size"] = len(self.reinvent_trainer.replay_buffer)
            status["diversity_filter_size"] = len(self.reinvent_trainer.diversity_filter)

        return status

    async def initialize(self):
        logger.info("Initializing generator engine...")

        # Initialize scoring pipeline first (always needed)
        # Don't overwrite if already set (allows pre-configuration with data sources)
        if self.scoring is None:
            self.scoring = ScoringPipeline()
        await self.scoring.initialize()

        # Try to load pretrained model if configured
        if self.config.use_pretrained:
            logger.info(f"Attempting to load pretrained model: {self.config.pretrained_model_name}")
            self.pretrained_generator = PretrainedMoleculeGenerator(
                model_name=self.config.pretrained_model_name,
                device=self.config.device
            )
            success = await self.pretrained_generator.initialize()

            if success:
                self._use_pretrained_for_generation = True
                # Use pretrained tokenizer for encoding/decoding
                logger.info("Using pretrained model for generation")
            else:
                logger.warning("Pretrained model failed to load, falling back to random policy")
                self._use_pretrained_for_generation = False

        # Always initialize the fallback policy (needed for RL training)
        self.tokenizer = SMIAISTokenizer()
        self.policy = MoleculePolicy(
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_ff=self.config.d_ff,
            max_len=self.config.max_len,
            pad_token_id=self.tokenizer.pad_token_id
        )
        self.policy.to(self.config.device)
        self.optimizer = PolicyOptimizer(self.policy, lr=self.config.learning_rate)

        if not self._use_pretrained_for_generation:
            logger.info("Using random policy for generation")

        # Initialize REINVENT trainer if RL fine-tuning is enabled
        if self.config.use_rl_finetuning and self._use_pretrained_for_generation:
            await self._setup_reinvent_trainer()

        # Setup molecular filters
        filter_config = FilterConfig(
            min_mw=150.0,
            max_mw=800.0,
            require_quaternary_nitrogen=True,
            apply_pains_filter=True,
            apply_brenk_filter=True,
        )
        self.molecular_filter = MolecularFilter(filter_config)
        self.diversity_selector = DiversitySelector(similarity_threshold=0.7)
        logger.info("Molecular filters initialized")

        logger.info("Generator engine ready")
    
    async def shutdown(self):
        logger.info("Generator engine shut down")

    async def _setup_reinvent_trainer(self):
        """Initialize REINVENT trainer with pretrained model as prior"""
        if not self.pretrained_generator or not self.pretrained_generator.is_ready:
            logger.warning("Cannot setup REINVENT trainer: pretrained model not ready")
            return

        logger.info("Setting up REINVENT trainer for RL fine-tuning...")

        reinvent_config = ReinventConfig(
            learning_rate=self.config.learning_rate,
            batch_size=64,
            sigma=self.config.rl_sigma,
            replay_buffer_size=self.config.rl_replay_buffer_size,
            diversity_filter=self.config.rl_diversity_filter,
            similarity_threshold=self.config.rl_similarity_threshold,
            early_stop_patience=self.config.rl_early_stop_patience,
            kl_weight=self.config.rl_kl_weight,
        )

        self.reinvent_trainer = ReinventTrainer(
            prior_model=self.pretrained_generator.model,
            tokenizer=self.pretrained_generator.tokenizer,
            config=reinvent_config,
            device=self.config.device,
        )

        logger.info("REINVENT trainer initialized successfully")

    async def start_rl_training(
        self,
        scoring_function: Callable[[List[str]], List[float]],
        max_steps: int = 1000,
        callback: Optional[Callable[[TrainingMetrics], None]] = None
    ):
        """Start REINVENT RL fine-tuning loop"""
        if not self.reinvent_trainer:
            raise ValueError("REINVENT trainer not initialized")

        if self.rl_training_active:
            raise ValueError("RL training already active")

        self.rl_training_active = True
        self._stop_requested = False

        logger.info(f"Starting REINVENT RL training for {max_steps} steps")

        try:
            for step in range(max_steps):
                if self._stop_requested:
                    logger.info("RL training stopped by request")
                    break

                # Run one training step
                metrics = self.reinvent_trainer.train_step(scoring_function)
                self._rl_metrics = metrics

                # Call callback if provided
                if callback:
                    callback(metrics)

                # Log progress every 10 steps
                if step % 10 == 0:
                    logger.info(
                        f"RL Step {step}: score={metrics.mean_score:.2f}, "
                        f"valid={metrics.valid_ratio:.2%}, unique={metrics.unique_ratio:.2%}"
                    )

                # Check for early stopping
                if self.reinvent_trainer._check_early_stopping():
                    logger.info(f"Early stopping at step {step}")
                    break

                await asyncio.sleep(0.01)  # Yield to event loop

        finally:
            self.rl_training_active = False
            logger.info("REINVENT RL training completed")

        return self._rl_metrics

    async def stop_rl_training(self):
        """Stop ongoing RL training"""
        if not self.rl_training_active:
            return
        self._stop_requested = True
        while self.rl_training_active:
            await asyncio.sleep(0.1)

    def create_rl_scoring_function(self, constraints: dict, weights: dict) -> Callable[[List[str]], List[float]]:
        """Create a scoring function for REINVENT RL training"""
        quat_constraints = QuatConstraints(**constraints)

        def scoring_function(smiles_list: List[str]) -> List[float]:
            scores = []
            for smiles in smiles_list:
                try:
                    is_valid, mol = validate_quat(smiles, quat_constraints)
                    if not is_valid:
                        scores.append(0.0)
                        continue

                    # Synchronous scoring for RL loop
                    score = self.score_molecule(smiles)
                    combined = (
                        weights.get("efficacy", 0.4) * score.get("efficacy", 0) +
                        weights.get("safety", 0.3) * score.get("safety", 0) +
                        weights.get("environmental", 0.2) * score.get("environmental", 0) +
                        weights.get("sa_score", 0.1) * score.get("sa_score", 0)
                    )
                    scores.append(combined)
                except Exception:
                    scores.append(0.0)

            return scores

        return scoring_function

    async def run_rl_generation(
        self,
        num_molecules: int,
        constraints: dict,
        weights: dict,
        rl_steps: int = 100,
        batch_size: int = 64
    ):
        """Run generation with REINVENT RL fine-tuning"""
        if not self.reinvent_trainer:
            logger.warning("REINVENT trainer not available, falling back to standard generation")
            return await self.run_generation(num_molecules, constraints, weights, batch_size)

        logger.info(f"Starting RL-guided generation: {num_molecules} molecules, {rl_steps} RL steps")

        self.state = GenerationState(
            is_running=True,
            start_time=time.time(),
            total_batches=rl_steps
        )
        self._stop_requested = False

        scoring_function = self.create_rl_scoring_function(constraints, weights)
        quat_constraints = QuatConstraints(**constraints)

        try:
            for step in range(rl_steps):
                if self._stop_requested or self.state.molecules_valid >= num_molecules:
                    break

                # Run RL training step
                metrics = self.reinvent_trainer.train_step(scoring_function)
                self._rl_metrics = metrics

                # Generate molecules using fine-tuned agent
                molecules = self.reinvent_trainer._generate_batch(batch_size)

                # Score and store valid molecules
                valid_molecules, scores = await self._score_batch(molecules, quat_constraints)

                if valid_molecules:
                    await self._store_molecules(valid_molecules, scores)
                    await self._update_pareto()
                    self._update_best_scores(scores)

                self.state.current_batch = step + 1
                self.state.molecules_generated += len(molecules)
                self.state.molecules_valid += len(valid_molecules)

                await asyncio.sleep(0.01)

        finally:
            self.state.is_running = False
            logger.info(f"RL generation complete: {self.state.molecules_valid} valid molecules")

    async def run_generation(self, num_molecules: int, constraints: dict, weights: dict,
                            batch_size: int = 64, use_gpu: bool = True, num_workers: int = 8):
        logger.info(f"Starting generation: target={num_molecules}, batch={batch_size}")
        self.state = GenerationState(is_running=True, start_time=time.time(),
                                     total_batches=num_molecules // batch_size + 1)
        self._stop_requested = False
        quat_constraints = QuatConstraints(**constraints)
        
        try:
            while self.state.molecules_valid < num_molecules and not self._stop_requested:
                molecules = await self._generate_batch(batch_size)
                valid_molecules, scores = await self._score_batch(molecules, quat_constraints)
                rewards = self._calculate_rewards(scores, weights)
                if valid_molecules:
                    await self._update_policy(valid_molecules, rewards)
                await self._store_molecules(valid_molecules, scores)
                await self._update_pareto()
                self.state.current_batch += 1
                self.state.molecules_generated += len(molecules)
                self.state.molecules_valid += len(valid_molecules)
                if scores:
                    self._update_best_scores(scores)
                await asyncio.sleep(0.01)
        finally:
            self.state.is_running = False
            logger.info(f"Generation complete: {self.state.molecules_valid} valid molecules")
    
    async def stop(self):
        self._stop_requested = True
        while self.state.is_running:
            await asyncio.sleep(0.1)
    
    async def _generate_batch(self, batch_size: int) -> List[str]:
        if self._use_pretrained_for_generation and self.pretrained_generator is not None:
            # Use pretrained model for generation
            return self.pretrained_generator.generate(
                batch_size=batch_size,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                max_length=self.config.max_len
            )
        else:
            # Use random policy as fallback
            self.policy.eval()
            with torch.no_grad():
                sequences, _ = self.policy.generate(
                    batch_size=batch_size,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    device=self.config.device
                )
            return [self.tokenizer.decode(seq.tolist()) for seq in sequences]
    
    async def _score_batch(self, smiles_list: List[str], constraints: QuatConstraints):
        valid_smiles, scores = [], []
        for smiles in smiles_list:
            is_valid, mol = validate_quat(smiles, constraints)
            if not is_valid:
                continue
            try:
                score = await self.scoring.score_molecule(smiles)
                valid_smiles.append(smiles)
                scores.append(score)
            except Exception:
                continue
        return valid_smiles, scores
    
    def _calculate_rewards(self, scores: List[dict], weights: dict) -> np.ndarray:
        return np.array([(weights["efficacy"] * s["efficacy"] + weights["safety"] * s["safety"] +
                         weights["environmental"] * s["environmental"] + weights["sa_score"] * s["sa_score"]) / 100
                        for s in scores])
    
    async def _update_policy(self, smiles_list: List[str], rewards: np.ndarray):
        self.policy.train()
        sequences = torch.tensor([self.tokenizer.encode(s, max_length=self.config.max_len) for s in smiles_list],
                                device=self.config.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.config.device)
        logits, values = self.policy(sequences)
        probs = torch.softmax(logits[:, :-1], dim=-1)
        actions = sequences[:, 1:]
        log_probs = torch.log(probs.gather(2, actions.unsqueeze(-1)).squeeze(-1) + 1e-10)
        policy_loss, value_loss, _ = self.policy.compute_loss_rl(sequences, log_probs, rewards_tensor, values)
        self.optimizer.step(policy_loss, value_loss)
    
    async def _store_molecules(self, smiles_list: List[str], scores: List[dict]):
        async with get_db_context() as db:
            for smiles, score in zip(smiles_list, scores):
                if await queries.get_molecule_by_smiles(db, smiles):
                    continue
                await queries.create_molecule(db, {
                    "smiles": smiles, "efficacy_score": score["efficacy"], "safety_score": score["safety"],
                    "environmental_score": score["environmental"], "sa_score": score["sa_score"],
                    "combined_score": (score["efficacy"] + score["safety"] + score["environmental"] + score["sa_score"]) / 4,
                    "molecular_weight": score.get("mw"), "logp": score.get("logp"),
                    "chain_length": score.get("chain_length"), "is_valid_quat": True
                })
    
    async def _update_pareto(self):
        async with get_db_context() as db:
            pareto_count = await queries.update_pareto_frontier(db)
            self.state.pareto_frontier = [{"count": pareto_count}]
    
    def _update_best_scores(self, scores: List[dict]):
        for s in scores:
            self.state.best_efficacy = max(self.state.best_efficacy, s["efficacy"])
            self.state.best_safety = max(self.state.best_safety, s["safety"])
            self.state.best_environmental = max(self.state.best_environmental, s["environmental"])
            combined = (s["efficacy"] + s["safety"] + s["environmental"] + s["sa_score"]) / 4
            self.state.best_combined = max(self.state.best_combined, combined)
    
    async def reset(self):
        self.state = GenerationState()
    
    def get_config(self) -> dict:
        config = {
            "model_path": "models/policy.pt",
            "vocab_size": self.vocab_size,
            "hidden_size": self.config.d_model,
            "num_layers": self.config.n_layers,
            "learning_rate": self.config.learning_rate,
            "temperature": self.config.temperature,
            "use_pretrained": self.config.use_pretrained,
            "using_pretrained": self._use_pretrained_for_generation,
        }
        if self._use_pretrained_for_generation and self.pretrained_generator:
            config["pretrained_model"] = self.config.pretrained_model_name
            config["pretrained_vocab_size"] = self.pretrained_generator.vocab_size
        return config
    
    async def update_config(self, config: dict):
        for k, v in config.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
    
    async def get_pareto_frontier(self) -> List[dict]:
        async with get_db_context() as db:
            molecules, _ = await queries.get_molecules(db, limit=1000, filters={"pareto_only": True})
            return [m.to_dict() for m in molecules]
    
    def tokenize(self, smiles: str) -> List[int]:
        return self.tokenizer.encode(smiles)
    
    def generate_one(self) -> str:
        if self._use_pretrained_for_generation and self.pretrained_generator is not None:
            smiles_list = self.pretrained_generator.generate(
                batch_size=1,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                max_length=self.config.max_len
            )
            return smiles_list[0] if smiles_list else ""
        else:
            self.policy.eval()
            with torch.no_grad():
                sequences, _ = self.policy.generate(
                    batch_size=1,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    device=self.config.device
                )
            return self.tokenizer.decode(sequences[0].tolist())
    
    def score_molecule(self, smiles: str) -> dict:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.scoring.score_molecule(smiles))
        finally:
            loop.close()

    async def filter_molecules(self, smiles_list: List[str]) -> tuple:
        """Filter molecules and return passed ones with statistics"""
        if not self.molecular_filter:
            return smiles_list, {"filtered": 0, "passed": len(smiles_list)}

        passed, reports = self.molecular_filter.filter_batch(smiles_list)
        rejection_summary = self.molecular_filter.get_rejection_summary(reports)

        return passed, {
            "total": len(smiles_list),
            "passed": len(passed),
            "filtered": len(smiles_list) - len(passed),
            "rejection_reasons": rejection_summary
        }

    async def select_diverse(self, smiles_list: List[str], scores: List[float],
                            n_select: int = 100) -> List[tuple]:
        """Select diverse subset of molecules"""
        if not self.diversity_selector:
            return list(zip(smiles_list[:n_select], scores[:n_select]))

        return self.diversity_selector.select_diverse(smiles_list, scores, n_select)
