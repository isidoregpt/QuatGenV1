"""Quaternary Ammonium Constraints - Validation rules for generated molecules"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import re


@dataclass
class QuatConstraints:
    min_mw: float = 200.0
    max_mw: float = 600.0
    min_chain_length: int = 8
    max_chain_length: int = 18
    require_quat: bool = True
    min_quat_nitrogens: int = 1
    max_quat_nitrogens: int = 2
    allowed_counterions: List[str] = field(default_factory=lambda: ["Cl", "Br", "I"])
    require_novel: bool = True
    max_rotatable_bonds: int = 20
    max_rings: int = 4
    require_positive_charge: bool = True


def validate_smiles(smiles: str) -> Tuple[bool, Optional[object]]:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return (mol is not None, mol)
    except Exception:
        return False, None


def count_quaternary_nitrogens(mol) -> int:
    try:
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "N" and atom.GetFormalCharge() == 1 and atom.GetDegree() == 4:
                count += 1
        return count
    except Exception:
        return 0


def get_longest_chain(mol) -> int:
    try:
        return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C" and not a.GetIsAromatic())
    except Exception:
        return 0


def get_counterion(smiles: str) -> Optional[str]:
    patterns = [(r"\.\[Cl-\]", "Cl"), (r"\.\[Br-\]", "Br"), (r"\.\[I-\]", "I"),
                (r"\[Cl-\]", "Cl"), (r"\[Br-\]", "Br"), (r"\[I-\]", "I")]
    for pattern, ion in patterns:
        if re.search(pattern, smiles):
            return ion
    return None


def calculate_molecular_weight(mol) -> float:
    try:
        from rdkit.Chem import Descriptors
        return Descriptors.MolWt(mol)
    except Exception:
        return 0.0


def get_formal_charge(mol) -> int:
    try:
        return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    except Exception:
        return 0


def validate_quat(smiles: str, constraints: QuatConstraints) -> Tuple[bool, Optional[object]]:
    is_valid, mol = validate_smiles(smiles)
    if not is_valid or mol is None:
        return False, None
    
    mw = calculate_molecular_weight(mol)
    if mw < constraints.min_mw or mw > constraints.max_mw:
        return False, None
    
    if constraints.require_quat:
        num_quat = count_quaternary_nitrogens(mol)
        if num_quat < constraints.min_quat_nitrogens or num_quat > constraints.max_quat_nitrogens:
            return False, None
    
    chain = get_longest_chain(mol)
    if chain < constraints.min_chain_length or chain > constraints.max_chain_length:
        return False, None
    
    counterion = get_counterion(smiles)
    if counterion and counterion not in constraints.allowed_counterions:
        return False, None
    
    if constraints.require_positive_charge and get_formal_charge(mol) < 1:
        return False, None
    
    return True, mol
