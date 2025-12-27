"""
Generator Engine
Main RL-based molecule generation loop

Coordinates:
- Policy network for SMILES generation
- Scoring pipeline for reward calculation
- Database storage for generated molecules
- Pareto frontier tracking
"""

import asyncio
import time
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
import torch
import numpy as np

from generator.tokenizer import SMIAISTokenizer
from generator.policy import MoleculePolicy, PolicyOptimizer
from generator.constraints import QuatConstraints, validate_quat
from scoring.pipeline import ScoringPipeline
from database.connection import get_db_context
from database import queries

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for generation"""
    # Model parameters
    vocab_size: int = 500
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_len: int = 128
    
    # Generation parameters
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    
    # RL parameters
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_weight: float = 1.0
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 8


@dataclass
class GenerationState:
    """Mutable state during generation"""
    is_running: bool = False
    molecules_generated: int = 0
    molecules_valid: int = 0
    current_batch: int = 0
    total_batches: int = 0
    start_time: float = 0.0
    
    # Pareto tracking
    pareto_frontier: List[Dict] = field(default_factory=list)
    
    # Best scores seen
    best_efficacy: float = 0.0
    best_safety: float = 0.0
    best_environmental: float = 0.0
    best_combined: float = 0.0
    
    @property
    def elapsed_seconds(self) -> float:
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time
    
    @property
    def molecules_per_hour(self) -> float:
        if self.elapsed_seconds < 1:
            return 0.0
        return self.molecules_generated / self.elapsed_seconds * 3600
    
    @property
    def estimated_remaining_seconds(self) -> float:
        if self.molecules_per_hour < 1:
            return 0.0
        remaining = self.total_batches - self.current_batch
        return remaining * 64 / self.molecules_per_hour * 3600


class GeneratorEngine:
    """
    Main generator engine
    
    Manages the RL training loop for molecule generation.
    Generates SMILES strings, scores them, and updates the policy
    based on multi-objective rewards.
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self.state = GenerationState()
        
        # Components (initialized lazily)
        self.tokenizer: Optional[SMIAISTokenizer] = None
        self.policy: Optional[MoleculePolicy] = None
        self.optimizer: Optional[PolicyOptimizer] = None
        self.scoring: Optional[ScoringPipeline] = None
        
        # Process pool for parallel scoring
        self.executor: Optional[ProcessPoolExecutor] = None
        
        # Stop flag
        self._stop_requested = False
    
    @property
    def is_ready(self) -> bool:
        return self.policy is not None and self.scoring is not None
    
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
        return {
            "efficacy": self.state.best_efficacy,
            "safety": self.state.best_safety,
            "environmental": self.state.best_environmental,
            "combined": self.state.best_combined
        }
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing generator engine...")
        
        # Initialize tokenizer
        self.tokenizer = SMIAISTokenizer()
        logger.info(f"Tokenizer initialized with vocab size {self.tokenizer.vocab_size}")
        
        # Initialize policy network
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
        logger.info(f"Policy network initialized on {self.config.device}")
        
        # Initialize optimizer
        self.optimizer = PolicyOptimizer(
            self.policy,
            lr=self.config.learning_rate
        )
        
        # Initialize scoring pipeline
        self.scoring = ScoringPipeline()
        await self.scoring.initialize()
        logger.info("Scoring pipeline initialized")
        
        # Initialize process pool
        self.executor = ProcessPoolExecutor(max_workers=self.config.num_workers)
        
        logger.info("Generator engine ready")
    
    async def shutdown(self):
        """Clean up resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("Generator engine shut down")
    
    async def run_generation(
        self,
        num_molecules: int,
        constraints: dict,
        weights: dict,
        batch_size: int = 64,
        use_gpu: bool = True,
        num_workers: int = 8
    ):
        """
        Run the generation loop
        
        Args:
            num_molecules: Target number of valid molecules
            constraints: Quat constraints dictionary
            weights: Objective weights dictionary
            batch_size: Batch size for generation
            use_gpu: Whether to use GPU
            num_workers: Number of parallel workers for scoring
        """
        logger.info(f"Starting generation: target={num_molecules}, batch={batch_size}")
        
        # Update state
        self.state = GenerationState(
            is_running=True,
            start_time=time.time(),
            total_batches=num_molecules // batch_size + 1
        )
        self._stop_requested = False
        
        # Parse constraints
        quat_constraints = QuatConstraints(**constraints)
        
        try:
            while (self.state.molecules_valid < num_molecules and
                   not self._stop_requested):
                
                # Generate batch
                molecules = await self._generate_batch(batch_size)
                
                # Validate and score
                valid_molecules, scores = await self._score_batch(
                    molecules, quat_constraints
                )
                
                # Calculate rewards
                rewards = self._calculate_rewards(scores, weights)
                
                # Update policy
                if len(valid_molecules) > 0:
                    await self._update_policy(valid_molecules, rewards)
                
                # Store molecules
                await self._store_molecules(valid_molecules, scores)
                
                # Update Pareto frontier
                await self._update_pareto()
                
                # Update state
                self.state.current_batch += 1
                self.state.molecules_generated += len(molecules)
                self.state.molecules_valid += len(valid_molecules)
                
                # Update best scores
                if len(scores) > 0:
                    self._update_best_scores(scores)
                
                # Log progress
                if self.state.current_batch % 10 == 0:
                    logger.info(
                        f"Batch {self.state.current_batch}: "
                        f"{self.state.molecules_valid}/{num_molecules} valid, "
                        f"{self.state.molecules_per_hour:.0f}/hr"
                    )
                
                # Small delay to prevent blocking
                await asyncio.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
        
        finally:
            self.state.is_running = False
            logger.info(
                f"Generation complete: {self.state.molecules_valid} valid molecules, "
                f"{self.state.elapsed_seconds:.1f}s"
            )
    
    async def stop(self):
        """Request generation stop"""
        logger.info("Stop requested")
        self._stop_requested = True
        
        # Wait for current batch to complete
        while self.state.is_running:
            await asyncio.sleep(0.1)
    
    async def _generate_batch(self, batch_size: int) -> List[str]:
        """Generate a batch of SMILES strings"""
        self.policy.eval()
        
        with torch.no_grad():
            sequences, _ = self.policy.generate(
                batch_size=batch_size,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                device=self.config.device
            )
        
        # Decode to SMILES
        smiles_list = []
        for seq in sequences:
            smiles = self.tokenizer.decode(seq.tolist())
            smiles_list.append(smiles)
        
        return smiles_list
    
    async def _score_batch(
        self,
        smiles_list: List[str],
        constraints: QuatConstraints
    ) -> tuple[List[str], List[dict]]:
        """Score and filter a batch of molecules"""
        valid_smiles = []
        scores = []
        
        for smiles in smiles_list:
            # Validate SMILES and quat constraints
            is_valid, mol = validate_quat(smiles, constraints)
            
            if not is_valid or mol is None:
                continue
            
            # Score molecule
            try:
                score = await self.scoring.score_molecule(smiles)
                valid_smiles.append(smiles)
                scores.append(score)
            except Exception as e:
                logger.debug(f"Scoring failed for {smiles}: {e}")
                continue
        
        return valid_smiles, scores
    
    def _calculate_rewards(
        self,
        scores: List[dict],
        weights: dict
    ) -> np.ndarray:
        """Calculate weighted rewards from scores"""
        rewards = []
        
        for score in scores:
            reward = (
                weights["efficacy"] * score["efficacy"] +
                weights["safety"] * score["safety"] +
                weights["environmental"] * score["environmental"] +
                weights["sa_score"] * score["sa_score"]
            ) / 100.0  # Normalize to [0, 1]
            
            rewards.append(reward)
        
        return np.array(rewards)
    
    async def _update_policy(
        self,
        smiles_list: List[str],
        rewards: np.ndarray
    ):
        """Update policy with RL gradient"""
        if len(smiles_list) == 0:
            return
        
        self.policy.train()
        
        # Encode sequences
        sequences = []
        for smiles in smiles_list:
            ids = self.tokenizer.encode(smiles, max_length=self.config.max_len)
            sequences.append(ids)
        
        sequences = torch.tensor(sequences, device=self.config.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.config.device)
        
        # Forward pass to get log probs and values
        logits, values = self.policy(sequences)
        
        # Get log probs for taken actions
        probs = torch.softmax(logits[:, :-1], dim=-1)
        actions = sequences[:, 1:]
        log_probs = torch.log(
            probs.gather(2, actions.unsqueeze(-1)).squeeze(-1) + 1e-10
        )
        
        # Compute losses
        policy_loss, value_loss, metrics = self.policy.compute_loss_rl(
            sequences,
            log_probs,
            rewards_tensor,
            values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )
        
        # Optimization step
        self.optimizer.step(
            policy_loss,
            value_loss,
            policy_weight=self.config.policy_weight,
            value_weight=self.config.value_weight
        )
    
    async def _store_molecules(
        self,
        smiles_list: List[str],
        scores: List[dict]
    ):
        """Store molecules in database"""
        async with get_db_context() as db:
            for smiles, score in zip(smiles_list, scores):
                # Check for duplicates
                existing = await queries.get_molecule_by_smiles(db, smiles)
                if existing:
                    continue
                
                # Create molecule record
                molecule_data = {
                    "smiles": smiles,
                    "efficacy_score": score["efficacy"],
                    "safety_score": score["safety"],
                    "environmental_score": score["environmental"],
                    "sa_score": score["sa_score"],
                    "combined_score": (
                        score["efficacy"] + score["safety"] +
                        score["environmental"] + score["sa_score"]
                    ) / 4,
                    "molecular_weight": score.get("mw"),
                    "logp": score.get("logp"),
                    "chain_length": score.get("chain_length"),
                    "is_valid_quat": True,
                    "generation_step": self.state.molecules_generated
                }
                
                await queries.create_molecule(db, molecule_data)
    
    async def _update_pareto(self):
        """Update Pareto frontier in database"""
        async with get_db_context() as db:
            pareto_count = await queries.update_pareto_frontier(db)
            self.state.pareto_frontier = [{"count": pareto_count}]
    
    def _update_best_scores(self, scores: List[dict]):
        """Update best scores seen"""
        for score in scores:
            self.state.best_efficacy = max(
                self.state.best_efficacy, score["efficacy"]
            )
            self.state.best_safety = max(
                self.state.best_safety, score["safety"]
            )
            self.state.best_environmental = max(
                self.state.best_environmental, score["environmental"]
            )
            combined = (
                score["efficacy"] + score["safety"] +
                score["environmental"] + score["sa_score"]
            ) / 4
            self.state.best_combined = max(
                self.state.best_combined, combined
            )
    
    async def reset(self):
        """Reset generator state"""
        self.state = GenerationState()
        
        # Reinitialize policy
        if self.policy:
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
            
            self.optimizer = PolicyOptimizer(
                self.policy,
                lr=self.config.learning_rate
            )
    
    def get_config(self) -> dict:
        """Get current configuration"""
        return {
            "model_path": "models/policy.pt",
            "vocab_size": self.vocab_size,
            "hidden_size": self.config.d_model,
            "num_layers": self.config.n_layers,
            "learning_rate": self.config.learning_rate,
            "temperature": self.config.temperature
        }
    
    async def update_config(self, config: dict):
        """Update configuration"""
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    async def get_pareto_frontier(self) -> List[dict]:
        """Get current Pareto frontier molecules"""
        async with get_db_context() as db:
            molecules, _ = await queries.get_molecules(
                db,
                limit=1000,
                filters={"pareto_only": True}
            )
            return [m.to_dict() for m in molecules]
    
    def tokenize(self, smiles: str) -> List[int]:
        """Tokenize a SMILES string (for benchmarking)"""
        return self.tokenizer.encode(smiles)
    
    def generate_one(self) -> str:
        """Generate a single molecule (for benchmarking)"""
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
        """Score a single molecule (for benchmarking)"""
        # Synchronous wrapper for async scoring
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.scoring.score_molecule(smiles))
        finally:
            loop.close()
    
    async def generate_similar(self, smiles: str) -> tuple[str, dict]:
        """Generate a molecule similar to the input"""
        # Encode input
        input_ids = self.tokenizer.encode(smiles, max_length=self.config.max_len)
        
        # Use partial sequence as conditioning
        partial_len = len(input_ids) // 2
        partial = input_ids[:partial_len]
        
        # Complete the sequence
        self.policy.eval()
        with torch.no_grad():
            input_tensor = torch.tensor([partial], device=self.config.device)
            
            # Continue generation from partial
            for _ in range(self.config.max_len - partial_len):
                logits, _ = self.policy(input_tensor)
                next_logits = logits[:, -1, :] / self.config.temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                input_tensor = torch.cat([input_tensor, next_token], dim=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        new_smiles = self.tokenizer.decode(input_tensor[0].tolist())
        scores = await self.scoring.score_molecule(new_smiles)
        
        return new_smiles, scores
