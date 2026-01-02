"""
Synthetic Accessibility Scoring

Implements the Ertl & Schuffenhauer SA_Score algorithm plus additional
synthesizability metrics for quaternary ammonium compounds.

Reference: Ertl, P. & Schuffenhauer, A. J. Cheminform. 1, 8 (2009)
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pickle
import os

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import (
        Descriptors, rdMolDescriptors, AllChem,
        Fragments, GraphDescriptors
    )
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available - SA scoring will be limited")


@dataclass
class SAScoreResult:
    """Detailed synthetic accessibility score result"""
    sa_score: float                    # 1-10 scale (1=easy, 10=hard)
    normalized_score: float            # 0-100 scale (100=easy, 0=hard)
    fragment_score: float              # Contribution from fragment frequencies
    complexity_penalty: float          # Penalty for molecular complexity
    stereo_penalty: float              # Penalty for stereocenters
    macrocycle_penalty: float          # Penalty for macrocycles
    spiro_penalty: float               # Penalty for spiro centers
    bridged_penalty: float             # Penalty for bridged systems
    quat_synthesis_score: float        # Quat-specific synthesis assessment
    problematic_groups: List[str]      # Potentially problematic functional groups
    estimated_steps: int               # Estimated synthetic steps
    confidence: float                  # Confidence in the prediction


class SAScorer:
    """
    Synthetic Accessibility Scorer using the SA_Score algorithm
    with extensions for quaternary ammonium compounds.
    """

    # Fragment scores from SA_Score paper (subset - full set would be loaded from file)
    # Higher score = more common = easier to synthesize
    COMMON_FRAGMENTS = {
        # Common simple fragments
        "[CH3]": 1.0,
        "[CH2]": 1.0,
        "[OH]": 0.9,
        "[NH2]": 0.9,
        "[Cl]": 0.8,
        "[Br]": 0.7,
        "[F]": 0.8,
        "c1ccccc1": 0.9,      # Benzene
        "C(=O)O": 0.8,        # Carboxylic acid
        "C(=O)N": 0.8,        # Amide
        "[N+]": 0.7,          # Quaternary nitrogen (common in quats)
        # Less common
        "[Si]": 0.4,
        "[B]": 0.3,
        "[Se]": 0.2,
    }

    # Problematic functional groups for synthesis
    PROBLEMATIC_GROUPS = {
        "azide": "[N-]=[N+]=[N-]",
        "peroxide": "OO",
        "nitro_aromatic": "[c]N(=O)=O",
        "acyl_halide": "C(=O)[Cl,Br,I]",
        "isocyanate": "N=C=O",
        "isothiocyanate": "N=C=S",
        "azo": "N=N",
        "diazo": "[N-]=[N+]",
        "nitroso": "N=O",
        "enamine": "C=CN",
        "vinyl_ether": "C=CO",
        "gem_dihalide": "C([F,Cl,Br,I])([F,Cl,Br,I])",
        "epoxide": "C1OC1",
        "aziridine": "C1NC1",
    }

    # Quat-friendly synthesis routes
    QUAT_FRIENDLY_PATTERNS = {
        "alkyl_tertiary_amine": "[NX3;!$(NC=O)]",  # Tertiary amine (quat precursor)
        "alkyl_halide": "[CX4][Cl,Br,I]",          # For quaternization
        "pyridine": "n1ccccc1",                     # Pyridinium precursor
        "imidazole": "n1ccnc1",                     # Imidazolium precursor
    }

    def __init__(self):
        self._is_ready = False
        self.fragment_scores = {}

    async def initialize(self):
        """Initialize scorer, optionally loading fragment scores from file"""
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - using simplified scoring")
            self._is_ready = True
            return

        # Try to load fragment scores from file (from SA_Score paper data)
        fragment_file = "data/sa_score_fragments.pkl"
        if os.path.exists(fragment_file):
            try:
                with open(fragment_file, "rb") as f:
                    self.fragment_scores = pickle.load(f)
                logger.info(f"Loaded {len(self.fragment_scores)} fragment scores")
            except Exception as e:
                logger.warning(f"Could not load fragment scores: {e}")
                self.fragment_scores = self.COMMON_FRAGMENTS
        else:
            self.fragment_scores = self.COMMON_FRAGMENTS

        self._is_ready = True
        logger.info("SA scorer initialized")

    async def score(self, smiles: str) -> Dict:
        """
        Calculate synthetic accessibility score.

        Returns:
            Dict with score (0-100), components, and detailed analysis
        """
        if not RDKIT_AVAILABLE:
            return self._fallback_score(smiles)

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"score": 0, "components": {}, "error": "Invalid SMILES"}

            # Calculate SA_Score components
            result = self._calculate_sa_score(mol, smiles)

            # Convert to 0-100 scale (invert: lower SA = higher score)
            normalized = max(0, (10 - result.sa_score) / 9 * 100)

            # Build response
            components = {
                "sa_score_raw": round(result.sa_score, 2),
                "fragment_contribution": round(result.fragment_score, 2),
                "complexity_penalty": round(result.complexity_penalty, 2),
                "stereo_penalty": round(result.stereo_penalty, 2),
                "ring_penalty": round(result.macrocycle_penalty + result.spiro_penalty + result.bridged_penalty, 2),
                "quat_synthesis": round(result.quat_synthesis_score, 1),
                "estimated_steps": result.estimated_steps,
            }

            return {
                "score": round(normalized, 1),
                "components": components,
                "problematic_groups": result.problematic_groups,
                "confidence": round(result.confidence, 2)
            }

        except Exception as e:
            logger.error(f"SA scoring error: {e}")
            return {"score": 50, "components": {}, "error": str(e)}

    def _calculate_sa_score(self, mol, smiles: str) -> SAScoreResult:
        """Calculate full SA_Score with all components"""

        # 1. Fragment score (based on fragment frequencies in known compounds)
        fragment_score = self._calculate_fragment_score(mol)

        # 2. Complexity penalty
        complexity_penalty = self._calculate_complexity_penalty(mol)

        # 3. Stereochemistry penalty
        stereo_penalty = self._calculate_stereo_penalty(mol)

        # 4. Ring system penalties
        macrocycle_penalty = self._calculate_macrocycle_penalty(mol)
        spiro_penalty = self._calculate_spiro_penalty(mol)
        bridged_penalty = self._calculate_bridged_penalty(mol)

        # 5. Combine into SA_Score (1-10 scale)
        # Base score from fragments (lower = easier)
        sa_score = fragment_score

        # Add penalties
        sa_score += complexity_penalty
        sa_score += stereo_penalty
        sa_score += macrocycle_penalty
        sa_score += spiro_penalty
        sa_score += bridged_penalty

        # Clamp to 1-10 range
        sa_score = max(1.0, min(10.0, sa_score))

        # 6. Quat-specific synthesis assessment
        quat_synthesis_score = self._assess_quat_synthesis(mol, smiles)

        # 7. Identify problematic groups
        problematic_groups = self._identify_problematic_groups(mol)

        # 8. Estimate synthetic steps
        estimated_steps = self._estimate_synthetic_steps(mol)

        # 9. Calculate confidence
        confidence = self._calculate_confidence(mol)

        return SAScoreResult(
            sa_score=sa_score,
            normalized_score=max(0, (10 - sa_score) / 9 * 100),
            fragment_score=fragment_score,
            complexity_penalty=complexity_penalty,
            stereo_penalty=stereo_penalty,
            macrocycle_penalty=macrocycle_penalty,
            spiro_penalty=spiro_penalty,
            bridged_penalty=bridged_penalty,
            quat_synthesis_score=quat_synthesis_score,
            problematic_groups=problematic_groups,
            estimated_steps=estimated_steps,
            confidence=confidence
        )

    def _calculate_fragment_score(self, mol) -> float:
        """
        Calculate fragment contribution to SA score.
        Molecules with common fragments are easier to synthesize.
        """
        # Use Morgan fingerprint fragments as proxy
        try:
            # Get atom environments (similar to SA_Score approach)
            info = {}
            fp = AllChem.GetMorganFingerprint(mol, 2, bitInfo=info)

            # Count fragments and their "rarity"
            num_atoms = mol.GetNumHeavyAtoms()

            if num_atoms == 0:
                return 5.0

            # Simple heuristic: more complex fragments = higher score
            num_unique_environments = len(info)

            # Normalize: more unique environments relative to size = harder
            fragment_ratio = num_unique_environments / num_atoms

            # Scale to contribute ~1-4 to SA score
            return 1.0 + fragment_ratio * 3

        except Exception:
            return 3.0  # Default middle value

    def _calculate_complexity_penalty(self, mol) -> float:
        """Calculate penalty based on molecular complexity"""
        try:
            # Bertz complexity index
            complexity = GraphDescriptors.BertzCT(mol)

            # Normalize (typical drug-like molecules: 100-1000)
            if complexity < 100:
                return 0.0
            elif complexity < 500:
                return (complexity - 100) / 400 * 1.5
            elif complexity < 1000:
                return 1.5 + (complexity - 500) / 500 * 1.5
            else:
                return 3.0 + min(2.0, (complexity - 1000) / 1000)

        except Exception:
            return 1.0

    def _calculate_stereo_penalty(self, mol) -> float:
        """Calculate penalty for stereocenters"""
        try:
            # Count chiral centers
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            num_chiral = len(chiral_centers)

            # Count double bond stereochemistry
            num_db_stereo = sum(1 for bond in mol.GetBonds()
                                if bond.GetStereo() != Chem.BondStereo.STEREONONE)

            # Penalty: 0.5 per chiral center, 0.25 per stereo double bond
            return num_chiral * 0.5 + num_db_stereo * 0.25

        except Exception:
            return 0.0

    def _calculate_macrocycle_penalty(self, mol) -> float:
        """Calculate penalty for macrocycles (rings > 8 atoms)"""
        try:
            ring_info = mol.GetRingInfo()
            ring_sizes = [len(ring) for ring in ring_info.AtomRings()]

            macrocycles = [size for size in ring_sizes if size > 8]

            if not macrocycles:
                return 0.0

            # Larger macrocycles are harder
            return sum(0.3 * (size - 8) for size in macrocycles)

        except Exception:
            return 0.0

    def _calculate_spiro_penalty(self, mol) -> float:
        """Calculate penalty for spiro centers"""
        try:
            # Spiro atoms are in exactly 2 rings and share no bonds between rings
            ring_info = mol.GetRingInfo()
            atom_rings = ring_info.AtomRings()

            spiro_count = 0
            for atom_idx in range(mol.GetNumAtoms()):
                rings_containing = [ring for ring in atom_rings if atom_idx in ring]
                if len(rings_containing) >= 2:
                    # Check if it's a true spiro (only this atom shared)
                    ring1, ring2 = rings_containing[0], rings_containing[1]
                    shared = set(ring1) & set(ring2)
                    if len(shared) == 1:
                        spiro_count += 1

            return spiro_count * 0.75

        except Exception:
            return 0.0

    def _calculate_bridged_penalty(self, mol) -> float:
        """Calculate penalty for bridged ring systems"""
        try:
            ring_info = mol.GetRingInfo()
            atom_rings = ring_info.AtomRings()

            if len(atom_rings) < 2:
                return 0.0

            # Check for bridged systems (rings sharing > 2 atoms but not fused)
            bridged_count = 0
            for i, ring1 in enumerate(atom_rings):
                for ring2 in atom_rings[i+1:]:
                    shared = set(ring1) & set(ring2)
                    if 2 < len(shared) < min(len(ring1), len(ring2)):
                        bridged_count += 1

            return bridged_count * 1.0

        except Exception:
            return 0.0

    def _assess_quat_synthesis(self, mol, smiles: str) -> float:
        """
        Assess ease of quaternary ammonium synthesis specifically.
        Returns 0-100 score where higher = easier quat synthesis.
        """
        score = 50.0  # Default neutral

        try:
            # Check if already a quat
            has_quat = "[N+]" in smiles or "[n+]" in smiles

            if has_quat:
                # Assess the quat center
                score += 20  # Already quaternary is good

                # Simple alkyl quat (4 carbons attached)?
                # Pattern: [N+](C)(C)(C)C
                simple_alkyl_quat = Chem.MolFromSmarts("[N+;X4]([CX4])([CX4])([CX4])[CX4]")
                if simple_alkyl_quat and mol.HasSubstructMatch(simple_alkyl_quat):
                    score += 15  # Simple to make via Menshutkin reaction

                # Pyridinium (easy)?
                pyridinium = Chem.MolFromSmarts("[n+]1ccccc1")
                if pyridinium and mol.HasSubstructMatch(pyridinium):
                    score += 15  # Easy quaternization of pyridine

                # Benzyl quat (common)?
                benzyl_quat = Chem.MolFromSmarts("[N+]Cc1ccccc1")
                if benzyl_quat and mol.HasSubstructMatch(benzyl_quat):
                    score += 10  # Benzyl halides react easily
            else:
                # Check for tertiary amine precursor
                tertiary_amine = Chem.MolFromSmarts("[NX3;!$(NC=O);!$(NS=O)]")
                if tertiary_amine and mol.HasSubstructMatch(tertiary_amine):
                    score += 10  # Has precursor for quaternization

            # Penalties for difficult quat synthesis
            # Multiple quat centers
            quat_pattern = Chem.MolFromSmarts("[N+,n+]")
            if quat_pattern:
                quat_matches = mol.GetSubstructMatches(quat_pattern)
                if len(quat_matches) > 1:
                    score -= 10 * (len(quat_matches) - 1)  # Multiple quats harder

            # Long alkyl chains (> C18) are harder to attach
            chain_carbons = sum(1 for a in mol.GetAtoms()
                                if a.GetSymbol() == "C" and not a.GetIsAromatic())
            if chain_carbons > 18:
                score -= (chain_carbons - 18) * 2

            return max(0, min(100, score))

        except Exception:
            return 50.0

    def _identify_problematic_groups(self, mol) -> List[str]:
        """Identify functional groups that complicate synthesis"""
        problematic = []

        try:
            for name, smarts in self.PROBLEMATIC_GROUPS.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    problematic.append(name)
        except Exception:
            pass

        return problematic

    def _estimate_synthetic_steps(self, mol) -> int:
        """Estimate number of synthetic steps required"""
        try:
            # Heuristic based on complexity
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            num_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            num_hetero = sum(1 for a in mol.GetAtoms() if a.GetSymbol() not in ["C", "H"])
            num_atoms = mol.GetNumHeavyAtoms()

            # Base steps
            steps = 2

            # Add for complexity
            steps += num_rings
            steps += num_chiral
            steps += max(0, num_hetero - 3)  # More than 3 heteroatoms adds steps
            steps += max(0, (num_atoms - 20) // 10)  # Larger molecules need more steps

            return min(20, steps)  # Cap at 20

        except Exception:
            return 5  # Default estimate

    def _calculate_confidence(self, mol) -> float:
        """Calculate confidence in the SA score prediction"""
        try:
            num_atoms = mol.GetNumHeavyAtoms()

            # Lower confidence for very small or very large molecules
            if num_atoms < 5:
                return 0.5
            elif num_atoms > 100:
                return 0.6

            # Check for unusual elements
            unusual_elements = {"Se", "Te", "As", "Sb", "Bi"}
            has_unusual = any(a.GetSymbol() in unusual_elements for a in mol.GetAtoms())

            if has_unusual:
                return 0.6

            return 0.85

        except Exception:
            return 0.5

    def _fallback_score(self, smiles: str) -> Dict:
        """Simple fallback when RDKit is not available"""
        # Very basic heuristics
        length = len(smiles)

        # Longer SMILES generally means more complex
        if length < 20:
            score = 80
        elif length < 50:
            score = 60
        elif length < 100:
            score = 40
        else:
            score = 20

        # Bonus for common quat patterns
        if "[N+]" in smiles:
            score += 10

        return {
            "score": min(100, score),
            "components": {"smiles_length_heuristic": length},
            "confidence": 0.3
        }

    @property
    def is_ready(self) -> bool:
        return self._is_ready
