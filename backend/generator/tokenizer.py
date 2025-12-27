"""
SMI+AIS Tokenizer
Hybrid SMILES + Atom-In-SMILES tokenization for molecular generation

Based on: "Improving Molecular Generation via Atomic-Level Information"
https://github.com/herim-han/AIS-Drug-Opt

AIS encoding captures local chemical environment:
- Central atom type
- Ring membership
- Neighboring atoms
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json
import os


@dataclass
class AISToken:
    """Atom-In-SMILES token with context"""
    atom: str           # Central atom (C, N, O, etc.)
    neighbors: str      # Sorted neighbor string (e.g., "CCN")
    in_ring: bool       # Is atom in a ring
    ring_sizes: tuple   # Sizes of rings containing this atom
    charge: int         # Formal charge
    
    def __str__(self):
        ring_str = "R" + "".join(str(s) for s in self.ring_sizes) if self.in_ring else ""
        charge_str = f"+{self.charge}" if self.charge > 0 else str(self.charge) if self.charge < 0 else ""
        return f"[{self.atom}{ring_str}:{self.neighbors}{charge_str}]"


class SMIAISTokenizer:
    """
    Hybrid SMI+AIS tokenizer
    
    Combines standard SMILES tokens with Atom-In-SMILES tokens that encode
    local chemical environment. This improves molecular generation by giving
    the model more context about each atom's chemical situation.
    
    Vocabulary structure:
    - Special tokens: [PAD], [UNK], [BOS], [EOS], [MASK]
    - SMILES tokens: atoms, bonds, brackets, numbers
    - AIS tokens: [atom:neighbors] with ring/charge info
    """
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    MASK_TOKEN = "[MASK]"
    
    # SMILES regex pattern
    SMILES_PATTERN = re.compile(
        r"(\[[^\]]+\]|"           # Bracketed atoms [Na+], [nH], etc.
        r"Br|Cl|"                 # Two-letter atoms
        r"[BCNOPSFIbcnops]|"      # Single-letter atoms
        r"[0-9]|"                 # Ring numbers
        r"[=@#%\.\-\+\(\)\\/])"  # Bonds and structure
    )
    
    def __init__(
        self,
        vocab_path: Optional[str] = None,
        max_ais_neighbors: int = 4,
        ais_vocab_size: int = 100
    ):
        """
        Initialize tokenizer
        
        Args:
            vocab_path: Path to vocabulary file (JSON)
            max_ais_neighbors: Max neighbors to encode in AIS tokens
            ais_vocab_size: Number of most common AIS patterns to include
        """
        self.max_ais_neighbors = max_ais_neighbors
        self.ais_vocab_size = ais_vocab_size
        
        # Initialize vocabularies
        self.smiles_vocab: Dict[str, int] = {}
        self.ais_vocab: Dict[str, int] = {}
        self.combined_vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        else:
            self._init_default_vocab()
    
    def _init_default_vocab(self):
        """Initialize default vocabulary"""
        # Special tokens
        special = [
            self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN,
            self.EOS_TOKEN, self.MASK_TOKEN
        ]
        
        # Basic SMILES tokens
        smiles_tokens = [
            # Organic atoms
            "B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I",
            "b", "c", "n", "o", "p", "s",
            # Bonds
            "-", "=", "#", ":", "/", "\\",
            # Structure
            "(", ")", "[", "]", ".", "+", "-",
            # Ring numbers
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "%10", "%11", "%12",
            # Common bracketed atoms
            "[H]", "[Na]", "[K]", "[Ca]", "[Mg]", "[Zn]", "[Fe]",
            "[N+]", "[N-]", "[O-]", "[S-]",
            "[nH]", "[NH]", "[NH2]", "[NH3+]",
            "[Cl-]", "[Br-]", "[I-]",
        ]
        
        # Build vocabulary
        idx = 0
        for token in special + smiles_tokens:
            self.combined_vocab[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        # Placeholder for AIS tokens (will be populated during training)
        self._ais_start_idx = idx
    
    def tokenize_smiles(self, smiles: str) -> List[str]:
        """Tokenize SMILES string into tokens"""
        tokens = self.SMILES_PATTERN.findall(smiles)
        return tokens
    
    def tokenize_ais(self, smiles: str) -> List[str]:
        """
        Tokenize SMILES with AIS encoding
        
        For each atom, creates an AIS token encoding its local environment.
        """
        try:
            from rdkit import Chem
        except ImportError:
            # Fallback to basic SMILES tokenization
            return self.tokenize_smiles(smiles)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self.tokenize_smiles(smiles)
        
        tokens = []
        ring_info = mol.GetRingInfo()
        
        for atom in mol.GetAtoms():
            # Get atom properties
            symbol = atom.GetSymbol()
            charge = atom.GetFormalCharge()
            
            # Get neighbors
            neighbors = sorted([n.GetSymbol() for n in atom.GetNeighbors()])
            neighbor_str = "".join(neighbors[:self.max_ais_neighbors])
            
            # Get ring info
            atom_rings = ring_info.AtomRings()
            atom_idx = atom.GetIdx()
            ring_sizes = tuple(sorted(
                len(ring) for ring in atom_rings if atom_idx in ring
            ))
            in_ring = len(ring_sizes) > 0
            
            # Create AIS token
            ais_token = AISToken(
                atom=symbol,
                neighbors=neighbor_str,
                in_ring=in_ring,
                ring_sizes=ring_sizes,
                charge=charge
            )
            
            token_str = str(ais_token)
            
            # Use AIS token if in vocab, otherwise use atom symbol
            if token_str in self.combined_vocab:
                tokens.append(token_str)
            else:
                tokens.append(symbol)
        
        return tokens
    
    def tokenize(self, smiles: str, use_ais: bool = True) -> List[str]:
        """
        Tokenize SMILES string
        
        Args:
            smiles: Input SMILES string
            use_ais: Whether to use AIS encoding (default True)
        
        Returns:
            List of tokens
        """
        if use_ais:
            return self.tokenize_ais(smiles)
        else:
            return self.tokenize_smiles(smiles)
    
    def encode(
        self,
        smiles: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        use_ais: bool = True
    ) -> List[int]:
        """
        Encode SMILES to token IDs
        
        Args:
            smiles: Input SMILES string
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum sequence length (pad/truncate)
            use_ais: Whether to use AIS encoding
        
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(smiles, use_ais=use_ais)
        
        if add_special_tokens:
            tokens = [self.BOS_TOKEN] + tokens + [self.EOS_TOKEN]
        
        # Convert to IDs
        ids = []
        for token in tokens:
            if token in self.combined_vocab:
                ids.append(self.combined_vocab[token])
            else:
                ids.append(self.combined_vocab[self.UNK_TOKEN])
        
        # Handle max_length
        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length]
            else:
                pad_id = self.combined_vocab[self.PAD_TOKEN]
                ids = ids + [pad_id] * (max_length - len(ids))
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to SMILES
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Skip PAD, BOS, EOS tokens
        
        Returns:
            SMILES string
        """
        special = {
            self.combined_vocab.get(self.PAD_TOKEN),
            self.combined_vocab.get(self.BOS_TOKEN),
            self.combined_vocab.get(self.EOS_TOKEN),
            self.combined_vocab.get(self.MASK_TOKEN),
        }
        
        tokens = []
        for idx in ids:
            if skip_special_tokens and idx in special:
                continue
            if idx in self.id_to_token:
                token = self.id_to_token[idx]
                # Convert AIS token back to atom symbol
                if token.startswith("[") and ":" in token:
                    # Extract atom from AIS token [C:CN] -> C
                    atom = token[1:].split(":")[0].split("R")[0]
                    tokens.append(atom)
                else:
                    tokens.append(token)
            else:
                tokens.append("")
        
        return "".join(tokens)
    
    def build_vocab_from_smiles(
        self,
        smiles_list: List[str],
        min_freq: int = 10
    ) -> None:
        """
        Build vocabulary from a list of SMILES
        
        Args:
            smiles_list: List of SMILES strings
            min_freq: Minimum frequency for AIS tokens
        """
        ais_counts = defaultdict(int)
        
        for smiles in smiles_list:
            try:
                tokens = self.tokenize_ais(smiles)
                for token in tokens:
                    if token.startswith("[") and ":" in token:
                        ais_counts[token] += 1
            except Exception:
                continue
        
        # Add top AIS tokens to vocabulary
        sorted_ais = sorted(
            ais_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.ais_vocab_size]
        
        idx = self._ais_start_idx
        for token, count in sorted_ais:
            if count >= min_freq:
                self.combined_vocab[token] = idx
                self.id_to_token[idx] = token
                self.ais_vocab[token] = idx
                idx += 1
    
    def save_vocab(self, path: str) -> None:
        """Save vocabulary to file"""
        data = {
            "combined_vocab": self.combined_vocab,
            "ais_vocab": self.ais_vocab,
            "smiles_vocab": self.smiles_vocab,
            "max_ais_neighbors": self.max_ais_neighbors,
            "ais_vocab_size": self.ais_vocab_size
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load_vocab(self, path: str) -> None:
        """Load vocabulary from file"""
        with open(path, "r") as f:
            data = json.load(f)
        
        self.combined_vocab = data["combined_vocab"]
        self.ais_vocab = data.get("ais_vocab", {})
        self.smiles_vocab = data.get("smiles_vocab", {})
        self.max_ais_neighbors = data.get("max_ais_neighbors", 4)
        self.ais_vocab_size = data.get("ais_vocab_size", 100)
        
        # Rebuild id_to_token
        self.id_to_token = {v: k for k, v in self.combined_vocab.items()}
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size"""
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


# Convenience function
def create_tokenizer(vocab_path: Optional[str] = None) -> SMIAISTokenizer:
    """Create a tokenizer instance"""
    return SMIAISTokenizer(vocab_path=vocab_path)
