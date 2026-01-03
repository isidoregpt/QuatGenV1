"""
Molecular Filters for Quaternary Ammonium Compound Generation

Provides validation, drug-likeness, PAINS, and diversity filtering
for generated molecules.
"""

import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem, Descriptors, rdMolDescriptors,
        FilterCatalog, rdfiltercatalog
    )
    from rdkit.Chem.FilterCatalog import FilterCatalogParams
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available - filtering will be limited")


class FilterResult(Enum):
    """Result of applying a filter"""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class FilterReport:
    """Detailed report from filtering a molecule"""
    smiles: str
    is_valid: bool
    passed_all: bool
    filter_results: Dict[str, FilterResult]
    rejection_reasons: List[str]
    warnings: List[str]
    properties: Dict[str, float]


@dataclass
class FilterConfig:
    """Configuration for molecular filters"""
    # Validity
    require_valid_smiles: bool = True
    require_sanitization: bool = True

    # Quat-specific
    require_quaternary_nitrogen: bool = True
    min_quat_nitrogens: int = 1
    max_quat_nitrogens: int = 3
    allowed_counterions: List[str] = field(default_factory=lambda: ["Cl", "Br", "I", "F"])

    # Property ranges
    min_mw: float = 150.0
    max_mw: float = 800.0
    min_logp: float = -2.0
    max_logp: float = 10.0
    max_hbd: int = 5
    max_hba: int = 10
    max_rotatable_bonds: int = 20
    max_tpsa: float = 200.0

    # Chain length (important for surfactant quats)
    min_chain_length: int = 6
    max_chain_length: int = 22

    # Structural
    max_rings: int = 6
    max_ring_size: int = 8
    max_stereocenters: int = 4

    # PAINS and toxicity
    apply_pains_filter: bool = True
    apply_brenk_filter: bool = True
    apply_nih_filter: bool = False

    # Drug-likeness rules
    apply_lipinski: bool = False  # Quats often violate Lipinski
    apply_veber: bool = True

    # Diversity
    diversity_threshold: float = 0.7  # Tanimoto similarity threshold


class MolecularFilter:
    """
    Comprehensive molecular filter for quaternary ammonium compounds.
    """

    # SMARTS patterns for quat nitrogen detection
    QUAT_PATTERNS = {
        "aliphatic_quat": "[N+;X4;!$([N+](=O)=O)]",  # Aliphatic quat (not nitro)
        "aromatic_quat": "[n+;X3]",                    # Aromatic (pyridinium)
        "any_quat": "[N+,n+;!$([N+](=O)=O)]",         # Any quaternary nitrogen
    }

    # Counterion patterns
    COUNTERION_PATTERNS = {
        "Cl": "[Cl-]",
        "Br": "[Br-]",
        "I": "[I-]",
        "F": "[F-]",
        "acetate": "CC(=O)[O-]",
        "sulfate": "[O-]S(=O)(=O)[O-]",
    }

    # Problematic substructures for quats
    QUAT_EXCLUSIONS = {
        "nitro": "[N+](=O)[O-]",           # Nitro group (not a real quat)
        "n_oxide": "[N+]([O-])",           # N-oxide
        "diazonium": "[N+]#N",             # Diazonium
        "azide": "[N-]=[N+]=[N-]",         # Azide
    }

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        self._pains_catalog = None
        self._brenk_catalog = None
        self._setup_filter_catalogs()

    def _setup_filter_catalogs(self):
        """Setup RDKit filter catalogs for PAINS, etc."""
        if not RDKIT_AVAILABLE:
            return

        try:
            # PAINS filter
            if self.config.apply_pains_filter:
                params = FilterCatalogParams()
                params.AddCatalog(rdfiltercatalog.FilterCatalogParams.FilterCatalogs.PAINS)
                self._pains_catalog = FilterCatalog.FilterCatalog(params)

            # Brenk filter (unwanted substructures)
            if self.config.apply_brenk_filter:
                params = FilterCatalogParams()
                params.AddCatalog(rdfiltercatalog.FilterCatalogParams.FilterCatalogs.BRENK)
                self._brenk_catalog = FilterCatalog.FilterCatalog(params)

        except Exception as e:
            logger.warning(f"Could not setup filter catalogs: {e}")

    def filter_molecule(self, smiles: str) -> FilterReport:
        """
        Apply all filters to a molecule.

        Returns:
            FilterReport with detailed results
        """
        results = {}
        rejection_reasons = []
        warnings = []
        properties = {}

        # Basic validity
        mol = None
        if self.config.require_valid_smiles:
            mol, valid, reason = self._check_validity(smiles)
            results["validity"] = FilterResult.PASS if valid else FilterResult.FAIL
            if not valid:
                rejection_reasons.append(reason)
                return FilterReport(
                    smiles=smiles,
                    is_valid=False,
                    passed_all=False,
                    filter_results=results,
                    rejection_reasons=rejection_reasons,
                    warnings=warnings,
                    properties=properties
                )

        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return FilterReport(
                    smiles=smiles,
                    is_valid=False,
                    passed_all=False,
                    filter_results={"validity": FilterResult.FAIL},
                    rejection_reasons=["Invalid SMILES"],
                    warnings=[],
                    properties={}
                )

        # Calculate properties
        properties = self._calculate_properties(mol)

        # Quat-specific filters
        if self.config.require_quaternary_nitrogen:
            passed, reason = self._check_quaternary_nitrogen(mol, smiles)
            results["quaternary_nitrogen"] = FilterResult.PASS if passed else FilterResult.FAIL
            if not passed:
                rejection_reasons.append(reason)

        # Counterion check
        passed, reason = self._check_counterion(smiles)
        results["counterion"] = FilterResult.PASS if passed else FilterResult.WARN
        if not passed:
            warnings.append(reason)

        # Property range filters
        prop_passed, prop_reasons = self._check_property_ranges(properties)
        results["property_ranges"] = FilterResult.PASS if prop_passed else FilterResult.FAIL
        rejection_reasons.extend(prop_reasons)

        # Chain length (for surfactant quats)
        chain_passed, chain_reason = self._check_chain_length(mol)
        results["chain_length"] = FilterResult.PASS if chain_passed else FilterResult.WARN
        if not chain_passed:
            warnings.append(chain_reason)

        # Structural filters
        struct_passed, struct_reasons = self._check_structural_limits(mol)
        results["structural"] = FilterResult.PASS if struct_passed else FilterResult.FAIL
        rejection_reasons.extend(struct_reasons)

        # PAINS filter
        if self.config.apply_pains_filter and self._pains_catalog:
            pains_passed, pains_matches = self._check_pains(mol)
            results["pains"] = FilterResult.PASS if pains_passed else FilterResult.FAIL
            if not pains_passed:
                rejection_reasons.append(f"PAINS alert: {', '.join(pains_matches)}")

        # Brenk filter
        if self.config.apply_brenk_filter and self._brenk_catalog:
            brenk_passed, brenk_matches = self._check_brenk(mol)
            results["brenk"] = FilterResult.PASS if brenk_passed else FilterResult.WARN
            if not brenk_passed:
                warnings.append(f"Brenk alert: {', '.join(brenk_matches)}")

        # Drug-likeness
        if self.config.apply_veber:
            veber_passed = self._check_veber(properties)
            results["veber"] = FilterResult.PASS if veber_passed else FilterResult.WARN
            if not veber_passed:
                warnings.append("Fails Veber rules (may have poor oral bioavailability)")

        # Check for exclusion patterns
        excl_passed, excl_reason = self._check_exclusions(mol)
        results["exclusions"] = FilterResult.PASS if excl_passed else FilterResult.FAIL
        if not excl_passed:
            rejection_reasons.append(excl_reason)

        # Determine overall result
        is_valid = all(r != FilterResult.FAIL for r in results.values())
        passed_all = all(r == FilterResult.PASS for r in results.values())

        return FilterReport(
            smiles=smiles,
            is_valid=is_valid,
            passed_all=passed_all,
            filter_results=results,
            rejection_reasons=rejection_reasons,
            warnings=warnings,
            properties=properties
        )

    def _check_validity(self, smiles: str) -> Tuple[Optional[object], bool, str]:
        """Check if SMILES is valid and can be sanitized"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None, False, "Invalid SMILES syntax"

            if self.config.require_sanitization:
                try:
                    Chem.SanitizeMol(mol)
                except Exception as e:
                    return None, False, f"Sanitization failed: {str(e)}"

            return mol, True, ""

        except Exception as e:
            return None, False, f"Parsing error: {str(e)}"

    def _check_quaternary_nitrogen(self, mol, smiles: str) -> Tuple[bool, str]:
        """Check for quaternary nitrogen presence"""
        quat_pattern = Chem.MolFromSmarts(self.QUAT_PATTERNS["any_quat"])

        if quat_pattern is None:
            # Fallback to string check
            has_quat = "[N+]" in smiles or "[n+]" in smiles
            if not has_quat:
                return False, "No quaternary nitrogen found"
            return True, ""

        matches = mol.GetSubstructMatches(quat_pattern)
        num_quats = len(matches)

        if num_quats < self.config.min_quat_nitrogens:
            return False, f"Too few quaternary nitrogens ({num_quats} < {self.config.min_quat_nitrogens})"

        if num_quats > self.config.max_quat_nitrogens:
            return False, f"Too many quaternary nitrogens ({num_quats} > {self.config.max_quat_nitrogens})"

        return True, ""

    def _check_counterion(self, smiles: str) -> Tuple[bool, str]:
        """Check for appropriate counterion"""
        # Check if molecule has a counterion
        has_counterion = False
        found_counterion = None

        for ion_name, pattern in self.COUNTERION_PATTERNS.items():
            if pattern.replace("[", "").replace("]", "").lower() in smiles.lower() or \
               pattern in smiles:
                has_counterion = True
                found_counterion = ion_name
                break

        # Also check for disconnected anions
        if "." in smiles:
            parts = smiles.split(".")
            for part in parts:
                if "-]" in part and len(part) < 10:  # Short anion
                    has_counterion = True

        if not has_counterion:
            return False, "No counterion detected (may be implicit)"

        if found_counterion and found_counterion not in self.config.allowed_counterions:
            return False, f"Counterion {found_counterion} not in allowed list"

        return True, ""

    def _calculate_properties(self, mol) -> Dict[str, float]:
        """Calculate molecular properties"""
        props = {}

        try:
            props["mw"] = Descriptors.MolWt(mol)
            props["logp"] = Descriptors.MolLogP(mol)
            props["hbd"] = rdMolDescriptors.CalcNumHBD(mol)
            props["hba"] = rdMolDescriptors.CalcNumHBA(mol)
            props["tpsa"] = Descriptors.TPSA(mol)
            props["rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
            props["num_rings"] = rdMolDescriptors.CalcNumRings(mol)
            props["num_aromatic_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
            props["num_heavy_atoms"] = mol.GetNumHeavyAtoms()
            props["formal_charge"] = Chem.GetFormalCharge(mol)

            # Stereocenters
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            props["num_stereocenters"] = len(chiral_centers)

            # Fraction sp3 carbons
            props["fsp3"] = rdMolDescriptors.CalcFractionCSP3(mol)

        except Exception as e:
            logger.warning(f"Property calculation error: {e}")

        return props

    def _check_property_ranges(self, props: Dict) -> Tuple[bool, List[str]]:
        """Check if properties are within allowed ranges"""
        reasons = []

        checks = [
            ("mw", self.config.min_mw, self.config.max_mw, "Molecular weight"),
            ("logp", self.config.min_logp, self.config.max_logp, "LogP"),
            ("hbd", 0, self.config.max_hbd, "H-bond donors"),
            ("hba", 0, self.config.max_hba, "H-bond acceptors"),
            ("tpsa", 0, self.config.max_tpsa, "TPSA"),
            ("rotatable_bonds", 0, self.config.max_rotatable_bonds, "Rotatable bonds"),
        ]

        for prop, min_val, max_val, name in checks:
            if prop in props:
                val = props[prop]
                if val < min_val:
                    reasons.append(f"{name} too low ({val:.1f} < {min_val})")
                elif val > max_val:
                    reasons.append(f"{name} too high ({val:.1f} > {max_val})")

        return len(reasons) == 0, reasons

    def _check_chain_length(self, mol) -> Tuple[bool, str]:
        """Check alkyl chain length (important for surfactant activity)"""
        # Count longest chain of aliphatic carbons
        try:
            # Simple heuristic: count non-aromatic, non-ring carbons
            chain_carbons = sum(
                1 for atom in mol.GetAtoms()
                if atom.GetSymbol() == "C" and not atom.GetIsAromatic() and not atom.IsInRing()
            )

            if chain_carbons < self.config.min_chain_length:
                return False, f"Chain too short ({chain_carbons} < {self.config.min_chain_length})"

            if chain_carbons > self.config.max_chain_length:
                return False, f"Chain too long ({chain_carbons} > {self.config.max_chain_length})"

            return True, ""

        except Exception:
            return True, ""  # Pass if can't determine

    def _check_structural_limits(self, mol) -> Tuple[bool, List[str]]:
        """Check structural complexity limits"""
        reasons = []

        # Ring count
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        if num_rings > self.config.max_rings:
            reasons.append(f"Too many rings ({num_rings} > {self.config.max_rings})")

        # Ring sizes
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if len(ring) > self.config.max_ring_size:
                reasons.append(f"Ring too large ({len(ring)} > {self.config.max_ring_size})")
                break

        # Stereocenters
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        if len(chiral) > self.config.max_stereocenters:
            reasons.append(f"Too many stereocenters ({len(chiral)} > {self.config.max_stereocenters})")

        return len(reasons) == 0, reasons

    def _check_pains(self, mol) -> Tuple[bool, List[str]]:
        """Check for PAINS (Pan Assay Interference Compounds) alerts"""
        if not self._pains_catalog:
            return True, []

        matches = self._pains_catalog.GetMatches(mol)
        if matches:
            match_names = [m.GetDescription() for m in matches]
            return False, match_names

        return True, []

    def _check_brenk(self, mol) -> Tuple[bool, List[str]]:
        """Check for Brenk unwanted substructures"""
        if not self._brenk_catalog:
            return True, []

        matches = self._brenk_catalog.GetMatches(mol)
        if matches:
            match_names = [m.GetDescription() for m in matches]
            return False, match_names

        return True, []

    def _check_veber(self, props: Dict) -> bool:
        """Check Veber rules for oral bioavailability"""
        # Veber: RotBonds <= 10 and TPSA <= 140
        rot_bonds = props.get("rotatable_bonds", 0)
        tpsa = props.get("tpsa", 0)

        return rot_bonds <= 10 and tpsa <= 140

    def _check_exclusions(self, mol) -> Tuple[bool, str]:
        """Check for excluded substructures"""
        for name, smarts in self.QUAT_EXCLUSIONS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return False, f"Contains excluded group: {name}"

        return True, ""

    def filter_batch(self, smiles_list: List[str]) -> Tuple[List[str], List[FilterReport]]:
        """
        Filter a batch of molecules.

        Returns:
            Tuple of (passed_smiles, all_reports)
        """
        passed = []
        reports = []

        for smiles in smiles_list:
            report = self.filter_molecule(smiles)
            reports.append(report)
            if report.is_valid:
                passed.append(smiles)

        return passed, reports

    def get_rejection_summary(self, reports: List[FilterReport]) -> Dict[str, int]:
        """Summarize rejection reasons from batch filtering"""
        summary = {}

        for report in reports:
            for reason in report.rejection_reasons:
                # Extract main reason category
                category = reason.split(":")[0].split("(")[0].strip()
                summary[category] = summary.get(category, 0) + 1

        return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))


class DiversitySelector:
    """
    Select diverse subset of molecules using various strategies.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self._fingerprint_cache: Dict[str, np.ndarray] = {}

    def _get_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """Get Morgan fingerprint for molecule"""
        if smiles in self._fingerprint_cache:
            return self._fingerprint_cache[smiles]

        if not RDKIT_AVAILABLE:
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_array = np.zeros((2048,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, fp_array)

            self._fingerprint_cache[smiles] = fp_array
            return fp_array

        except Exception:
            return None

    def _tanimoto_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Tanimoto similarity between fingerprints"""
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        return intersection / union if union > 0 else 0.0

    def select_diverse(self,
                       smiles_list: List[str],
                       scores: Optional[List[float]] = None,
                       n_select: Optional[int] = None,
                       strategy: str = "maxmin") -> List[Tuple[str, float]]:
        """
        Select diverse subset of molecules.

        Args:
            smiles_list: List of SMILES
            scores: Optional scores for each molecule
            n_select: Number to select (default: all that pass diversity)
            strategy: "maxmin" (maximize minimum distance) or "leader" (leader-picker)

        Returns:
            List of (smiles, score) tuples for selected molecules
        """
        if not smiles_list:
            return []

        if scores is None:
            scores = [1.0] * len(smiles_list)

        # Get fingerprints
        fps = []
        valid_indices = []
        for i, smiles in enumerate(smiles_list):
            fp = self._get_fingerprint(smiles)
            if fp is not None:
                fps.append(fp)
                valid_indices.append(i)

        if not fps:
            return []

        if strategy == "maxmin":
            selected_indices = self._maxmin_selection(fps, n_select)
        else:  # leader
            selected_indices = self._leader_selection(fps, n_select)

        # Map back to original indices and return
        results = []
        for idx in selected_indices:
            orig_idx = valid_indices[idx]
            results.append((smiles_list[orig_idx], scores[orig_idx]))

        return results

    def _maxmin_selection(self, fps: List[np.ndarray], n_select: Optional[int]) -> List[int]:
        """MaxMin diversity selection - maximize minimum distance to selected set"""
        n = len(fps)
        n_select = n_select or n

        if n <= n_select:
            return list(range(n))

        # Start with first molecule
        selected = [0]
        remaining = set(range(1, n))

        while len(selected) < n_select and remaining:
            # Find molecule with maximum minimum distance to selected set
            best_idx = None
            best_min_dist = -1

            for idx in remaining:
                min_dist = min(
                    1 - self._tanimoto_similarity(fps[idx], fps[sel_idx])
                    for sel_idx in selected
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx

            if best_idx is not None and best_min_dist > (1 - self.similarity_threshold):
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break  # No more diverse molecules

        return selected

    def _leader_selection(self, fps: List[np.ndarray], n_select: Optional[int]) -> List[int]:
        """Leader-picker selection - greedy clustering"""
        n = len(fps)
        n_select = n_select or n

        selected = []

        for i in range(n):
            if len(selected) >= n_select:
                break

            # Check if similar to any selected molecule
            is_diverse = True
            for sel_idx in selected:
                sim = self._tanimoto_similarity(fps[i], fps[sel_idx])
                if sim > self.similarity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(i)

        return selected

    def calculate_diversity_metrics(self, smiles_list: List[str]) -> Dict[str, float]:
        """Calculate diversity metrics for a set of molecules"""
        if len(smiles_list) < 2:
            return {"mean_diversity": 1.0, "min_diversity": 1.0, "max_diversity": 1.0}

        fps = [self._get_fingerprint(s) for s in smiles_list]
        fps = [fp for fp in fps if fp is not None]

        if len(fps) < 2:
            return {"mean_diversity": 1.0, "min_diversity": 1.0, "max_diversity": 1.0}

        # Calculate all pairwise distances
        distances = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = self._tanimoto_similarity(fps[i], fps[j])
                distances.append(1 - sim)

        return {
            "mean_diversity": float(np.mean(distances)),
            "min_diversity": float(np.min(distances)),
            "max_diversity": float(np.max(distances)),
            "std_diversity": float(np.std(distances)),
            "num_molecules": len(fps)
        }

    def clear_cache(self):
        """Clear fingerprint cache"""
        self._fingerprint_cache.clear()

    def __len__(self) -> int:
        """Return number of cached fingerprints"""
        return len(self._fingerprint_cache)
