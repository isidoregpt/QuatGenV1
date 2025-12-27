"""SMI+AIS Tokenizer - Hybrid SMILES + Atom-In-SMILES tokenization"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import json
import os


@dataclass
class AISToken:
    atom: str
    neighbors: str
    in_ring: bool
    ring_sizes: tuple
    charge: int
    
    def __str__(self):
        ring_str = "R" + "".join(str(s) for s in self.ring_sizes) if self.in_ring else ""
        charge_str = f"+{self.charge}" if self.charge > 0 else str(self.charge) if self.charge < 0 else ""
        return f"[{self.atom}{ring_str}:{self.neighbors}{charge_str}]"


class SMIAISTokenizer:
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    MASK_TOKEN = "[MASK]"
    
    SMILES_PATTERN = re.compile(r"(\[[^\]]+\]|Br|Cl|[BCNOPSFIbcnops]|[0-9]|[=@#%\.\-\+\(\)\\/])")
    
    def __init__(self, vocab_path: Optional[str] = None, max_ais_neighbors: int = 4, ais_vocab_size: int = 100):
        self.max_ais_neighbors = max_ais_neighbors
        self.ais_vocab_size = ais_vocab_size
        self.smiles_vocab: Dict[str, int] = {}
        self.ais_vocab: Dict[str, int] = {}
        self.combined_vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        else:
            self._init_default_vocab()
    
    def _init_default_vocab(self):
        special = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.MASK_TOKEN]
        smiles_tokens = ["B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "b", "c", "n", "o", "p", "s",
                        "-", "=", "#", ":", "/", "\\", "(", ")", "[", "]", ".", "+",
                        "1", "2", "3", "4", "5", "6", "7", "8", "9",
                        "[H]", "[Na]", "[K]", "[N+]", "[N-]", "[O-]", "[nH]", "[NH]", "[Cl-]", "[Br-]"]
        idx = 0
        for token in special + smiles_tokens:
            self.combined_vocab[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        self._ais_start_idx = idx
    
    def tokenize_smiles(self, smiles: str) -> List[str]:
        return self.SMILES_PATTERN.findall(smiles)
    
    def tokenize(self, smiles: str, use_ais: bool = False) -> List[str]:
        return self.tokenize_smiles(smiles)
    
    def encode(self, smiles: str, add_special_tokens: bool = True, max_length: Optional[int] = None, use_ais: bool = False) -> List[int]:
        tokens = self.tokenize(smiles, use_ais=use_ais)
        if add_special_tokens:
            tokens = [self.BOS_TOKEN] + tokens + [self.EOS_TOKEN]
        ids = [self.combined_vocab.get(t, self.combined_vocab[self.UNK_TOKEN]) for t in tokens]
        if max_length:
            if len(ids) > max_length:
                ids = ids[:max_length]
            else:
                ids = ids + [self.combined_vocab[self.PAD_TOKEN]] * (max_length - len(ids))
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        special = {self.combined_vocab.get(t) for t in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.MASK_TOKEN]}
        tokens = []
        for idx in ids:
            if skip_special_tokens and idx in special:
                continue
            token = self.id_to_token.get(idx, "")
            if token.startswith("[") and ":" in token:
                token = token[1:].split(":")[0].split("R")[0]
            tokens.append(token)
        return "".join(tokens)
    
    def build_vocab_from_smiles(self, smiles_list: List[str], min_freq: int = 10) -> None:
        pass  # Simplified - would build AIS vocab from data
    
    def save_vocab(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"combined_vocab": self.combined_vocab, "ais_vocab": self.ais_vocab}, f, indent=2)
    
    def load_vocab(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        self.combined_vocab = data["combined_vocab"]
        self.ais_vocab = data.get("ais_vocab", {})
        self.id_to_token = {v: k for k, v in self.combined_vocab.items()}
    
    @property
    def vocab_size(self) -> int:
        return len(self.combined_vocab)
    
    @property
    def pad_token_id(self) -> int:
        return self.combined_vocab[self.PAD_TOKEN]
    
    @property
    def bos_token_id(self) -> int:
        return self.combined_vocab[self.BOS_TOKEN]
    
    @property
    def eos_token_id(self) -> int:
        return self.combined_vocab[self.EOS_TOKEN]
    
    @property
    def unk_token_id(self) -> int:
        return self.combined_vocab[self.UNK_TOKEN]
