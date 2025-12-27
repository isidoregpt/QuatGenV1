"""AIS vocabulary builder and loader."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Iterable
from collections import Counter
import json
from pathlib import Path

from .ais import count_ais_frequencies


@dataclass
class AISVocab:
    top_n: int
    ais_tokens: List[str]
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]

    @staticmethod
    def build(smiles_iter: Iterable[str], top_n: int) -> "AISVocab":
        freq: Counter = count_ais_frequencies(smiles_iter)
        most_common = [t for t, _ in freq.most_common(top_n)]
        return AISVocab(top_n=top_n, ais_tokens=most_common, token_to_id={}, id_to_token={})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {"top_n": self.top_n, "ais_tokens": self.ais_tokens}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "AISVocab":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return AISVocab(top_n=int(payload["top_n"]), ais_tokens=list(payload["ais_tokens"]), token_to_id={}, id_to_token={})
