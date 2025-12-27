"""
Quaternary Ammonium Constraints
Validation rules for generated molecules

A valid quaternary ammonium compound must have:
- At least one quaternary nitrogen (positively charged, 4 bonds)
- Appropriate molecular weight range
- Alkyl chain(s) of suitable length
- Acceptable counterion
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import re


@dataclass
class QuatConstraints:
    """Constraints for valid quaternary ammonium compounds"""
    
    # Molecular weight
    min_mw: float = 200.0
    max_mw: float = 600.0
    
    # Alkyl chain length
    min_chain_length: int = 8
    max_chain_length: int = 18
    
    # Required features
    require_quat: bool = True
    min_quat_nitrogens: int = 1
    max_quat_nitrogens: int = 2
    
    # Allowed counterions
    allowed_counterions: List[str] = field(default_factory=lambda: ["Cl", "Br", "I"])
    
    # Novelty
    require_novel: bool = True
    
    # Additional filters
    max_rotatable_bonds: int = 20
    max_rings: int = 4
    require_positive_charge: bool = True


def validate_smiles(smiles: str) -> Tuple[bool, Optional[object]]:
    """
    Validate SMILES string and return RDKit molecule
    
    Returns:
        (is_valid, mol) tuple
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None
        return True, mol
    except Exception:
        return False, None


def count_quaternary_nitrogens(mol) -> int:
    """Count quaternary nitrogen atoms in molecule"""
    try:
        from rdkit import Chem
        
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "N":
                # Quaternary = 4 bonds and positive charge
                if atom.GetFormalCharge() == 1 and atom.GetDegree() == 4:
                    count += 1
                # Also count [N+] with 4 neighbors
                elif atom.GetFormalCharge() == 1 and len(list(atom.GetNeighbors())) == 4:
                    count += 1
        return count
    except Exception:
        return 0


def get_longest_chain(mol) -> int:
    """Get length of longest carbon chain"""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        
        # Simple approach: count carbons in longest path
        # More sophisticated: use SMARTS to find alkyl chains
        
        # Find all carbon atoms
        carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"]
        
        if not carbons:
            return 0
        
        # Use molecular descriptors for chain length
        # This is an approximation
        num_aliphatic_carbons = sum(
            1 for a in mol.GetAtoms()
            if a.GetSymbol() == "C" and not a.GetIsAromatic()
        )
        
        return num_aliphatic_carbons
    except Exception:
        return 0


def get_counterion(smiles: str) -> Optional[str]:
    """Extract counterion from SMILES (if ionic)"""
    # Common patterns
    counterion_patterns = [
        (r"\.\[Cl-\]", "Cl"),
        (r"\.\[Br-\]", "Br"),
        (r"\.\[I-\]", "I"),
        (r"\[Cl-\]", "Cl"),
        (r"\[Br-\]", "Br"),
        (r"\[I-\]", "I"),
    ]
    
    for pattern, ion in counterion_patterns:
        if re.search(pattern, smiles):
            return ion
    
    return None


def calculate_molecular_weight(mol) -> float:
    """Calculate molecular weight"""
    try:
        from rdkit.Chem import Descriptors
        return Descriptors.MolWt(mol)
    except Exception:
        return 0.0


def count_rotatable_bonds(mol) -> int:
    """Count rotatable bonds"""
    try:
        from rdkit.Chem import rdMolDescriptors
        return rdMolDescriptors.CalcNumRotatableBonds(mol)
    except Exception:
        return 0


def count_rings(mol) -> int:
    """Count number of rings"""
    try:
        return mol.GetRingInfo().NumRings()
    except Exception:
        return 0


def get_formal_charge(mol) -> int:
    """Get total formal charge"""
    try:
        return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    except Exception:
        return 0


def validate_quat(
    smiles: str,
    constraints: QuatConstraints
) -> Tuple[bool, Optional[object]]:
    """
    Validate a SMILES string against quat constraints
    
    Returns:
        (is_valid, mol) tuple where mol is the RDKit molecule if valid
    """
    # Basic SMILES validation
    is_valid, mol = validate_smiles(smiles)
    if not is_valid or mol is None:
        return False, None
    
    # Molecular weight
    mw = calculate_molecular_weight(mol)
    if mw < constraints.min_mw or mw > constraints.max_mw:
        return False, None
    
    # Quaternary nitrogen check
    if constraints.require_quat:
        num_quat_n = count_quaternary_nitrogens(mol)
        if num_quat_n < constraints.min_quat_nitrogens:
            return False, None
        if num_quat_n > constraints.max_quat_nitrogens:
            return False, None
    
    # Chain length
    chain_length = get_longest_chain(mol)
    if chain_length < constraints.min_chain_length:
        return False, None
    if chain_length > constraints.max_chain_length:
        return False, None
    
    # Counterion check
    counterion = get_counterion(smiles)
    if counterion and counterion not in constraints.allowed_counterions:
        return False, None
    
    # Rotatable bonds
    rot_bonds = count_rotatable_bonds(mol)
    if rot_bonds > constraints.max_rotatable_bonds:
        return False, None
    
    # Rings
    num_rings = count_rings(mol)
    if num_rings > constraints.max_rings:
        return False, None
    
    # Positive charge
    if constraints.require_positive_charge:
        charge = get_formal_charge(mol)
        if charge < 1:
            return False, None
    
    return True, mol


def extract_quat_properties(smiles: str, mol) -> dict:
    """
    Extract quat-specific properties from a molecule
    
    Returns dictionary with:
    - num_quat_n: Number of quaternary nitrogens
    - chain_length: Longest alkyl chain
    - counterion: Counterion if present
    - mw: Molecular weight
    - charge: Total formal charge
    """
    return {
        "num_quat_n": count_quaternary_nitrogens(mol),
        "chain_length": get_longest_chain(mol),
        "counterion": get_counterion(smiles),
        "mw": calculate_molecular_weight(mol),
        "charge": get_formal_charge(mol),
        "rotatable_bonds": count_rotatable_bonds(mol),
        "rings": count_rings(mol)
    }


# SMARTS patterns for common quat scaffolds
QUAT_SCAFFOLDS = {
    "benzalkonium": "[N+;X4](C)(C)(C)Cc1ccccc1",  # BAC
    "cetrimide": "[N+;X4](C)(C)(C)CCCCCCCCCCCCCCCC",  # CTAB
    "didecyl": "[N+;X4](C)(C)(CCCCCCCCCC)CCCCCCCCCC",  # DDAC
    "pyridinium": "[n+]1ccccc1",  # Pyridinium
}


def identify_scaffold(mol) -> Optional[str]:
    """Identify which scaffold type the molecule matches"""
    try:
        from rdkit import Chem
        
        for name, smarts in QUAT_SCAFFOLDS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return name
        
        return None
    except Exception:
        return None


def is_novel(smiles: str, known_smiles: set) -> bool:
    """Check if molecule is novel (not in known set)"""
    try:
        from rdkit import Chem
        
        # Canonicalize for comparison
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return canonical not in known_smiles
    except Exception:
        return False
