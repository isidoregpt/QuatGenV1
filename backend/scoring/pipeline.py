"""Scoring Pipeline - Multi-objective scoring for quaternary ammonium compounds"""

import asyncio
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    use_gpu: bool = True
    batch_size: int = 32
    cache_scores: bool = True


class ScoringPipeline:
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()
        self.efficacy_scorer = None
        self.safety_scorer = None
        self.environmental_scorer = None
        self.sa_scorer = None
        self._is_ready = False
    
    @property
    def is_ready(self) -> bool:
        return self._is_ready
    
    async def initialize(self):
        logger.info("Initializing scoring pipeline...")
        from scoring.efficacy import EfficacyScorer
        from scoring.safety import SafetyScorer
        from scoring.environmental import EnvironmentalScorer
        from scoring.sa_score import SAScorer
        
        self.efficacy_scorer = EfficacyScorer()
        await self.efficacy_scorer.initialize()
        self.safety_scorer = SafetyScorer()
        await self.safety_scorer.initialize()
        self.environmental_scorer = EnvironmentalScorer()
        await self.environmental_scorer.initialize()
        self.sa_scorer = SAScorer()
        await self.sa_scorer.initialize()
        self._is_ready = True
        logger.info("Scoring pipeline ready")
    
    async def score_molecule(self, smiles: str) -> Dict:
        if not self._is_ready:
            raise RuntimeError("Scoring pipeline not initialized")
        efficacy_result, safety_result, env_result, sa_result = await asyncio.gather(
            self.efficacy_scorer.score(smiles), self.safety_scorer.score(smiles),
            self.environmental_scorer.score(smiles), self.sa_scorer.score(smiles)
        )
        return {
            "efficacy": efficacy_result["score"], "safety": safety_result["score"],
            "environmental": env_result["score"], "sa_score": sa_result["score"],
            "mw": efficacy_result.get("mw"), "logp": efficacy_result.get("logp"),
            "chain_length": efficacy_result.get("chain_length")
        }
    
    async def score_batch(self, smiles_list: List[str]) -> List[Dict]:
        return await asyncio.gather(*[self.score_molecule(s) for s in smiles_list], return_exceptions=True)
