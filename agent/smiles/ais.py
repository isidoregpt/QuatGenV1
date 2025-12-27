"""
AIS token utilities (Atom-in-SMILES).

AIS token includes:
- central atom
- ring membership
- neighbor atom symbols

This implements a pragmatic AIS tokenization used for modeling.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
from collections import Counter

from rdkit import Chem


@dataclass(frozen=True)
class AISToken:
    s: str
    def __str__(self) -> str:
        return self.s


def _is_ring_atom(atom: Chem.Atom) -> bool:
    return bool(atom.IsInRing())


def _neighbor_symbols(atom: Chem.Atom) -> str:
    neigh = [n.GetSymbol() for n in atom.GetNeighbors()]
    neigh.sort()
    return "".join(neigh) if neigh else ""


def _central_symbol(atom: Chem.Atom) -> str:
    sym = atom.GetSymbol()
    if atom.GetIsAromatic():
        sym = sym.lower()
        if atom.GetTotalNumHs() > 0:
            sym = f"{sym}H"
    else:
        if sym == "C" and atom.GetTotalNumHs() == 3 and atom.GetDegree() == 1:
            sym = "CH3"
    return sym


def atom_to_ais_token(atom: Chem.Atom) -> AISToken:
    central = _central_symbol(atom)
    ring = "R" if _is_ring_atom(atom) else "!R"
    neigh = _neighbor_symbols(atom)
    return AISToken(f"[{central};{ring};{neigh}]")


def mol_to_ais_tokens(mol: Chem.Mol) -> List[str]:
    return [str(atom_to_ais_token(a)) for a in mol.GetAtoms()]


def count_ais_frequencies(smiles_iter) -> Counter:
    freq = Counter()
    for smi in smiles_iter:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        freq.update(mol_to_ais_tokens(mol))
    return freq
