"""
Tokenization for:
- plain SMILES
- SMI+AIS(N) hybrid

We keep a lightweight SMILES lexer and replace atom tokens with AIS tokens when present in vocab.
"""

from __future__ import annotations
from typing import List, Set

from rdkit import Chem
from .ais import atom_to_ais_token

_TWO_CHAR = {"Cl", "Br"}
_SPECIAL = set("()[]=#@+-\\/.: ")
_DIGITS = set("0123456789")


def tokenize_smiles_basic(smiles: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    while i < len(smiles):
        ch = smiles[i]
        if ch == "[":
            j = i + 1
            while j < len(smiles) and smiles[j] != "]":
                j += 1
            if j >= len(smiles):
                tokens.append(ch); i += 1; continue
            tokens.append(smiles[i:j+1]); i = j + 1; continue

        if i + 1 < len(smiles) and smiles[i:i+2] in _TWO_CHAR:
            tokens.append(smiles[i:i+2]); i += 2; continue

        if ch == "%":
            if i + 2 < len(smiles):
                tokens.append(smiles[i:i+3]); i += 3
            else:
                tokens.append(ch); i += 1
            continue

        if ch in _DIGITS:
            tokens.append(ch); i += 1; continue

        if ch in _SPECIAL:
            if ch != " ":
                tokens.append(ch)
            i += 1; continue

        if ch.isalpha():
            tokens.append(ch); i += 1; continue

        tokens.append(ch); i += 1
    return tokens


def tokenize_smi_ais(smiles: str, ais_token_set: Set[str]) -> List[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return tokenize_smiles_basic(smiles)

    can = Chem.MolToSmiles(mol, canonical=True)
    base_tokens = tokenize_smiles_basic(can)
    ais_by_atom = [str(atom_to_ais_token(a)) for a in mol.GetAtoms()]

    replaced: List[str] = []
    atom_cursor = 0
    for tok in base_tokens:
        is_atom_token = (tok.startswith("[") and tok.endswith("]")) or tok.isalpha() or tok in _TWO_CHAR
        if is_atom_token and atom_cursor < len(ais_by_atom):
            ais_tok = ais_by_atom[atom_cursor]
            replaced.append(ais_tok if ais_tok in ais_token_set else tok)
            atom_cursor += 1
        else:
            replaced.append(tok)
    return replaced
