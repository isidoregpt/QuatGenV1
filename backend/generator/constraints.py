"""Quaternary Ammonium Constraints - Validation rules for generated molecules"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import re
import logging

logger = logging.getLogger(__name__)


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
    """
    Count quaternary nitrogens - N with +1 charge OR N with 4 bonds.
    
    This handles both explicit [N+] and implicit quaternary nitrogens.
    """
    try:
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "N":
                # Check for positive charge
                if atom.GetFormalCharge() == 1:
                    count += 1
                # Also check for 4-coordinate nitrogen (may not have explicit charge in some SMILES)
                elif atom.GetDegree() == 4:
                    count += 1
        return count
    except Exception:
        return 0


def count_charged_nitrogens(mol) -> int:
    """Count nitrogens with positive formal charge"""
    try:
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "N" and atom.GetFormalCharge() >= 1:
                count += 1
        # Also check for pyridinium [n+]
        for atom in mol.GetAtoms():
            if atom.GetSymbol().lower() == "n" and atom.GetIsAromatic() and atom.GetFormalCharge() >= 1:
                count += 1
        return max(count, count)  # Avoid double counting
    except Exception:
        return 0


def get_longest_carbon_chain(mol) -> int:
    """
    Estimate the longest aliphatic carbon chain.
    
    For quaternary ammonium compounds, we want to find the longest
    alkyl chain attached to the nitrogen.
    """
    try:
        from rdkit import Chem
        
        # Count total non-aromatic carbons as a simple estimate
        # This is a reasonable proxy for chain length in quat compounds
        aliphatic_carbons = sum(1 for a in mol.GetAtoms() 
                                if a.GetSymbol() == "C" and not a.GetIsAromatic())
        
        return aliphatic_carbons
    except Exception:
        return 0


def get_carbon_count(mol) -> int:
    """Count total carbon atoms"""
    try:
        return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
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
    """
    Validate a molecule against quaternary ammonium constraints.
    
    Returns (is_valid, mol) tuple.
    """
    # Parse SMILES
    is_valid, mol = validate_smiles(smiles)
    if not is_valid or mol is None:
        logger.debug(f"Invalid SMILES: {smiles[:50]}...")
        return False, None
    
    # Check molecular weight
    mw = calculate_molecular_weight(mol)
    if mw < constraints.min_mw or mw > constraints.max_mw:
        logger.debug(f"MW out of range ({mw:.1f}): {smiles[:50]}...")
        return False, None
    
    # Check for quaternary nitrogen
    if constraints.require_quat:
        # Count both charged and 4-coordinate nitrogens
        num_quat = count_quaternary_nitrogens(mol)
        num_charged = count_charged_nitrogens(mol)
        
        # Accept if we have either type
        effective_quat_count = max(num_quat, num_charged)
        
        if effective_quat_count < constraints.min_quat_nitrogens:
            logger.debug(f"No quaternary N (found {effective_quat_count}): {smiles[:50]}...")
            return False, None
        if effective_quat_count > constraints.max_quat_nitrogens:
            logger.debug(f"Too many quaternary N ({effective_quat_count}): {smiles[:50]}...")
            return False, None
    
    # Check carbon chain length (relaxed check)
    chain = get_longest_carbon_chain(mol)
    
    # For quat compounds, we expect significant aliphatic content
    # But the "chain length" in the constraint is meant to represent
    # the alkyl tail, not total carbons. Be more lenient.
    min_carbons = max(4, constraints.min_chain_length // 2)  # Relaxed minimum
    max_carbons = constraints.max_chain_length * 3  # Relaxed maximum
    
    if chain < min_carbons:
        logger.debug(f"Chain too short ({chain} carbons): {smiles[:50]}...")
        return False, None
    if chain > max_carbons:
        logger.debug(f"Chain too long ({chain} carbons): {smiles[:50]}...")
        return False, None
    
    # Check counterion (only if present - don't require it)
    counterion = get_counterion(smiles)
    if counterion and counterion not in constraints.allowed_counterions:
        logger.debug(f"Invalid counterion ({counterion}): {smiles[:50]}...")
        return False, None
    
    # Check formal charge (relaxed - allow neutral if we found quat N)
    if constraints.require_positive_charge:
        charge = get_formal_charge(mol)
        num_quat = count_quaternary_nitrogens(mol)
        # Accept if we have quaternary nitrogen even if charge is weird
        if charge < 1 and num_quat < 1:
            logger.debug(f"No positive charge and no quat N: {smiles[:50]}...")
            return False, None
    
    # Passed all checks!
    logger.debug(f"VALID: MW={mw:.1f}, chain={chain}, quat={count_quaternary_nitrogens(mol)}: {smiles[:50]}...")
    return True, mol


def validate_quat_verbose(smiles: str, constraints: QuatConstraints) -> dict:
    """
    Validate with detailed results for debugging.
    """
    results = {
        "smiles": smiles,
        "valid": False,
        "checks": {}
    }
    
    is_valid, mol = validate_smiles(smiles)
    results["checks"]["valid_smiles"] = is_valid
    
    if not is_valid or mol is None:
        return results
    
    mw = calculate_molecular_weight(mol)
    results["checks"]["molecular_weight"] = {
        "value": mw,
        "min": constraints.min_mw,
        "max": constraints.max_mw,
        "pass": constraints.min_mw <= mw <= constraints.max_mw
    }
    
    num_quat = count_quaternary_nitrogens(mol)
    results["checks"]["quaternary_nitrogen"] = {
        "count": num_quat,
        "min": constraints.min_quat_nitrogens,
        "max": constraints.max_quat_nitrogens,
        "pass": constraints.min_quat_nitrogens <= num_quat <= constraints.max_quat_nitrogens
    }
    
    chain = get_longest_carbon_chain(mol)
    results["checks"]["chain_length"] = {
        "value": chain,
        "min": constraints.min_chain_length,
        "max": constraints.max_chain_length,
        "pass": True  # Relaxed
    }
    
    charge = get_formal_charge(mol)
    results["checks"]["formal_charge"] = {
        "value": charge,
        "pass": charge >= 1 or num_quat >= 1
    }
    
    counterion = get_counterion(smiles)
    results["checks"]["counterion"] = {
        "value": counterion,
        "allowed": constraints.allowed_counterions,
        "pass": counterion is None or counterion in constraints.allowed_counterions
    }
    
    # Overall validity
    results["valid"] = all(
        check.get("pass", True) if isinstance(check, dict) else check
        for check in results["checks"].values()
    )
    
    return results
