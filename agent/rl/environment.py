from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Set

from rdkit import Chem

from agent.smiles.vocab import AISVocab
from agent.smiles.tokenize import tokenize_smiles_basic, tokenize_smi_ais


@dataclass
class EnvConfig:
    representation_mode: str = "smiles"
    ais_vocab_path: str | None = None


class VirtualLabEnv:
    def __init__(self, cfg: EnvConfig, predictors, rules_engine):
        self.cfg = cfg
        self.predictors = predictors
        self.rules = rules_engine

        self.ais_set: Set[str] = set()
        if self.cfg.representation_mode == "smi+ais":
            if not self.cfg.ais_vocab_path:
                raise ValueError("smi+ais mode requires ais_vocab_path")
            vocab = AISVocab.load(self.cfg.ais_vocab_path)
            self.ais_set = set(vocab.ais_tokens)

    def tokenize_for_model(self, smiles: str):
        if self.cfg.representation_mode == "smi+ais":
            return tokenize_smi_ais(smiles, self.ais_set)
        return tokenize_smiles_basic(smiles)

    def evaluate(self, smiles: str) -> Dict[str, Any]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"ok": False, "reason": "invalid_smiles", "reward": -1.0}

        rule_result = self.rules.check(mol)
        if not rule_result.get("hard_ok", True):
            return {"ok": False, "reason": "hard_rule_violation", "reward": -1.0, "rules": rule_result}

        scores = self.predictors.score(mol)

        reward = (
            0.40 * scores["efficacy"]
            + 0.25 * scores["human_safety"]
            + 0.25 * scores["environment"]
            + 0.10 * scores["synth_access"]
            - 1.00 * rule_result.get("soft_penalty", 0.0)
        )

        return {"ok": True, "reward": float(reward), "scores": scores, "rules": rule_result}
