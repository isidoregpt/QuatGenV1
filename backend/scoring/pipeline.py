"""
Scoring Pipeline
Multi-objective scoring for quaternary ammonium compounds

Scores:
1. Efficacy (antimicrobial activity)
2. Safety (human toxicity)
3. Environmental (biodegradability, aquatic toxicity)
4. Synthetic Accessibility
"""

import asyncio
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """Configuration for scoring models"""
    use_gpu: bool = True
    batch_size: int = 32
    cache_scores: bool = True
    
    # Model paths
    efficacy_model_path: str = "models/efficacy.pt"
    safety_model_path: str = "models/safety.pt"
    environmental_model_path: str = "models/environmental.pt"


class ScoringPipeline:
    """
    Multi-objective scoring pipeline
    
    Combines multiple scoring models to evaluate quaternary ammonium compounds
    across efficacy, safety, and environmental dimensions.
    """
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()
        
        # Scoring modules
        self.efficacy_scorer: Optional["EfficacyScorer"] = None
        self.safety_scorer: Optional["SafetyScorer"] = None
        self.environmental_scorer: Optional["EnvironmentalScorer"] = None
        self.sa_scorer: Optional["SAScorer"] = None
        
        self._is_ready = False
    
    @property
    def is_ready(self) -> bool:
        return self._is_ready
    
    async def initialize(self):
        """Initialize all scoring models"""
        logger.info("Initializing scoring pipeline...")
        
        # Initialize efficacy scorer
        from scoring.efficacy import EfficacyScorer
        self.efficacy_scorer = EfficacyScorer()
        await self.efficacy_scorer.initialize()
        
        # Initialize safety scorer
        from scoring.safety import SafetyScorer
        self.safety_scorer = SafetyScorer()
        await self.safety_scorer.initialize()
        
        # Initialize environmental scorer
        from scoring.environmental import EnvironmentalScorer
        self.environmental_scorer = EnvironmentalScorer()
        await self.environmental_scorer.initialize()
        
        # Initialize SA scorer
        from scoring.sa_score import SAScorer
        self.sa_scorer = SAScorer()
        await self.sa_scorer.initialize()
        
        self._is_ready = True
        logger.info("Scoring pipeline ready")
    
    async def score_molecule(self, smiles: str) -> Dict:
        """
        Score a single molecule
        
        Returns dictionary with:
        - efficacy: 0-100 score
        - safety: 0-100 score
        - environmental: 0-100 score
        - sa_score: 0-100 score
        - details: Nested dict with component scores
        """
        if not self._is_ready:
            raise RuntimeError("Scoring pipeline not initialized")
        
        # Run all scorers in parallel
        efficacy_task = asyncio.create_task(
            self.efficacy_scorer.score(smiles)
        )
        safety_task = asyncio.create_task(
            self.safety_scorer.score(smiles)
        )
        environmental_task = asyncio.create_task(
            self.environmental_scorer.score(smiles)
        )
        sa_task = asyncio.create_task(
            self.sa_scorer.score(smiles)
        )
        
        # Gather results
        efficacy_result, safety_result, env_result, sa_result = await asyncio.gather(
            efficacy_task, safety_task, environmental_task, sa_task
        )
        
        return {
            "efficacy": efficacy_result["score"],
            "safety": safety_result["score"],
            "environmental": env_result["score"],
            "sa_score": sa_result["score"],
            "mw": efficacy_result.get("mw"),
            "logp": efficacy_result.get("logp"),
            "chain_length": efficacy_result.get("chain_length"),
            "details": {
                "efficacy": efficacy_result.get("components", {}),
                "safety": safety_result.get("components", {}),
                "environmental": env_result.get("components", {}),
                "sa": sa_result.get("components", {})
            }
        }
    
    async def score_batch(self, smiles_list: List[str]) -> List[Dict]:
        """Score multiple molecules"""
        tasks = [self.score_molecule(s) for s in smiles_list]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_score_descriptions(self) -> Dict:
        """Get descriptions of all score components"""
        return {
            "efficacy": {
                "name": "Antimicrobial Efficacy",
                "range": "0-100",
                "components": [
                    "gram_positive_mic",
                    "gram_negative_mic",
                    "antifungal",
                    "cmc_score",
                    "membrane_disruption"
                ]
            },
            "safety": {
                "name": "Human Safety",
                "range": "0-100",
                "components": [
                    "acute_toxicity",
                    "skin_irritation",
                    "eye_irritation",
                    "respiratory"
                ]
            },
            "environmental": {
                "name": "Environmental Impact",
                "range": "0-100",
                "components": [
                    "biodegradability",
                    "aquatic_toxicity",
                    "bioaccumulation",
                    "persistence"
                ]
            },
            "sa_score": {
                "name": "Synthetic Accessibility",
                "range": "0-100",
                "components": [
                    "sa_score",
                    "complexity",
                    "starting_materials"
                ]
            }
        }
