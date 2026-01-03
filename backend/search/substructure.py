"""
Substructure Search Module

Provides SMARTS-based substructure searching for molecular databases.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit import RDLogger
    # Suppress RDKit warnings for invalid SMILES during search
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available - substructure search disabled")


class CommonPatterns:
    """Pre-defined SMARTS patterns for common structural motifs"""

    # Quaternary ammonium patterns
    QUAT_ALIPHATIC = "[N+;X4;!$([N+](=O)=O)]"  # Aliphatic quaternary nitrogen
    QUAT_AROMATIC = "[n+;X3]"  # Aromatic quaternary (pyridinium)
    QUAT_ANY = "[N+,n+;!$([N+](=O)=O)]"  # Any quaternary nitrogen

    # Chain patterns
    LONG_ALKYL_CHAIN = "[CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2]"  # C8+ chain
    MEDIUM_ALKYL_CHAIN = "[CH2][CH2][CH2][CH2][CH2][CH2]"  # C6+ chain

    # Functional groups
    BENZYL = "[CH2]c1ccccc1"  # Benzyl group
    PHENYL = "c1ccccc1"  # Phenyl ring
    HYDROXYL = "[OH]"  # Hydroxyl group
    ETHER = "[OD2]([#6])[#6]"  # Ether linkage
    ESTER = "[#6][CX3](=O)[OX2][#6]"  # Ester group
    AMIDE = "[NX3][CX3](=[OX1])[#6]"  # Amide group

    # Common disinfectant scaffolds
    BAC_CORE = "[N+](C)(C)Cc1ccccc1"  # Benzalkonium core
    DDAC_CORE = "[N+](C)(C)[CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2]"  # Didecyl core
    CETYL_CORE = "[N+](C)(C)(C)[CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2]"  # Cetyl core

    # Halogen patterns
    CHLORO = "[Cl]"
    BROMO = "[Br]"
    FLUORO = "[F]"
    IODO = "[I]"

    # Ring systems
    PYRIDINIUM = "[n+]1ccccc1"
    IMIDAZOLIUM = "[n+]1ccnc1"
    MORPHOLINE = "C1COCCN1"

    @classmethod
    def get_all_patterns(cls) -> Dict[str, str]:
        """Get all defined patterns as a dictionary"""
        patterns = {}
        for name in dir(cls):
            if not name.startswith('_') and name.isupper():
                value = getattr(cls, name)
                if isinstance(value, str):
                    patterns[name] = value
        return patterns

    @classmethod
    def get_quat_patterns(cls) -> Dict[str, str]:
        """Get quaternary ammonium specific patterns"""
        return {
            "aliphatic_quat": cls.QUAT_ALIPHATIC,
            "aromatic_quat": cls.QUAT_AROMATIC,
            "any_quat": cls.QUAT_ANY,
            "bac_core": cls.BAC_CORE,
            "ddac_core": cls.DDAC_CORE,
            "cetyl_core": cls.CETYL_CORE,
        }


@dataclass
class SearchResult:
    """Result of a substructure search"""
    smiles: str
    molecule_id: Optional[int] = None
    name: Optional[str] = None
    match_atoms: List[Tuple[int, ...]] = field(default_factory=list)
    match_count: int = 0
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "smiles": self.smiles,
            "molecule_id": self.molecule_id,
            "name": self.name,
            "match_atoms": [list(m) for m in self.match_atoms],
            "match_count": self.match_count,
            "scores": self.scores,
            "metadata": self.metadata
        }


@dataclass
class SearchQuery:
    """Configuration for a substructure search"""
    pattern: str  # SMARTS pattern
    max_results: int = 100
    require_quat: bool = False  # Also require quaternary nitrogen
    min_efficacy: Optional[float] = None
    min_safety: Optional[float] = None
    min_sa: Optional[float] = None
    include_metadata: bool = True


class SubstructureSearch:
    """
    SMARTS-based substructure search engine.

    Searches molecules for structural patterns using RDKit's
    substructure matching capabilities.
    """

    def __init__(self):
        self._is_ready = RDKIT_AVAILABLE
        self._pattern_cache: Dict[str, Any] = {}

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def validate_smarts(self, smarts: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a SMARTS pattern.

        Args:
            smarts: SMARTS pattern string

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self._is_ready:
            return False, "RDKit not available"

        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                return False, "Invalid SMARTS syntax"
            return True, None
        except Exception as e:
            return False, str(e)

    def _get_pattern(self, smarts: str) -> Optional[Any]:
        """Get compiled pattern from cache or compile new"""
        if smarts in self._pattern_cache:
            return self._pattern_cache[smarts]

        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            self._pattern_cache[smarts] = pattern
        return pattern

    def search_molecule(self,
                        smiles: str,
                        smarts: str,
                        return_all_matches: bool = True) -> Optional[SearchResult]:
        """
        Search a single molecule for a SMARTS pattern.

        Args:
            smiles: SMILES string of molecule to search
            smarts: SMARTS pattern to search for
            return_all_matches: If True, return all matches; otherwise just first

        Returns:
            SearchResult if molecule matches, None otherwise
        """
        if not self._is_ready:
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            pattern = self._get_pattern(smarts)
            if pattern is None:
                return None

            if return_all_matches:
                matches = mol.GetSubstructMatches(pattern)
            else:
                match = mol.GetSubstructMatch(pattern)
                matches = (match,) if match else ()

            if not matches or not matches[0]:
                return None

            return SearchResult(
                smiles=smiles,
                match_atoms=list(matches),
                match_count=len(matches)
            )

        except Exception as e:
            logger.error(f"Search error for {smiles}: {e}")
            return None

    def search_molecules(self,
                         molecules: List[Dict[str, Any]],
                         query: SearchQuery) -> List[SearchResult]:
        """
        Search multiple molecules for a SMARTS pattern.

        Args:
            molecules: List of molecule dictionaries with 'smiles' key
            query: SearchQuery configuration

        Returns:
            List of SearchResult objects for matching molecules
        """
        if not self._is_ready:
            return []

        results = []
        pattern = self._get_pattern(query.pattern)
        if pattern is None:
            logger.warning(f"Invalid SMARTS pattern: {query.pattern}")
            return []

        # Pre-compile quat pattern if needed
        quat_pattern = None
        if query.require_quat:
            quat_pattern = self._get_pattern(CommonPatterns.QUAT_ANY)

        for mol_data in molecules:
            smiles = mol_data.get("smiles")
            if not smiles:
                continue

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                # Check pattern match
                matches = mol.GetSubstructMatches(pattern)
                if not matches:
                    continue

                # Check quat requirement
                if quat_pattern and not mol.HasSubstructMatch(quat_pattern):
                    continue

                # Check score filters
                scores = {}
                if "efficacy_score" in mol_data:
                    scores["efficacy"] = mol_data["efficacy_score"]
                    if query.min_efficacy and mol_data["efficacy_score"] < query.min_efficacy:
                        continue
                if "safety_score" in mol_data:
                    scores["safety"] = mol_data["safety_score"]
                    if query.min_safety and mol_data["safety_score"] < query.min_safety:
                        continue
                if "sa_score" in mol_data:
                    scores["sa"] = mol_data["sa_score"]
                    if query.min_sa and mol_data["sa_score"] < query.min_sa:
                        continue

                # Build result
                result = SearchResult(
                    smiles=smiles,
                    molecule_id=mol_data.get("id"),
                    name=mol_data.get("name"),
                    match_atoms=list(matches),
                    match_count=len(matches),
                    scores=scores
                )

                if query.include_metadata:
                    result.metadata = {
                        k: v for k, v in mol_data.items()
                        if k not in ["smiles", "id", "name", "efficacy_score",
                                     "safety_score", "sa_score"]
                    }

                results.append(result)

                if len(results) >= query.max_results:
                    break

            except Exception as e:
                logger.debug(f"Error searching molecule: {e}")
                continue

        return results

    def has_substructure(self, smiles: str, smarts: str) -> bool:
        """
        Quick check if molecule contains substructure.

        Args:
            smiles: SMILES string
            smarts: SMARTS pattern

        Returns:
            True if molecule contains pattern
        """
        if not self._is_ready:
            return False

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            pattern = self._get_pattern(smarts)
            if pattern is None:
                return False

            return mol.HasSubstructMatch(pattern)

        except Exception:
            return False

    def has_quat_nitrogen(self, smiles: str) -> bool:
        """Check if molecule has quaternary nitrogen"""
        return self.has_substructure(smiles, CommonPatterns.QUAT_ANY)

    def count_matches(self, smiles: str, smarts: str) -> int:
        """
        Count number of pattern matches in molecule.

        Args:
            smiles: SMILES string
            smarts: SMARTS pattern

        Returns:
            Number of matches
        """
        if not self._is_ready:
            return 0

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0

            pattern = self._get_pattern(smarts)
            if pattern is None:
                return 0

            matches = mol.GetSubstructMatches(pattern)
            return len(matches)

        except Exception:
            return 0

    def get_matched_atoms(self, smiles: str, smarts: str) -> List[Tuple[int, ...]]:
        """
        Get atom indices that match pattern.

        Args:
            smiles: SMILES string
            smarts: SMARTS pattern

        Returns:
            List of tuples of matching atom indices
        """
        result = self.search_molecule(smiles, smarts, return_all_matches=True)
        return result.match_atoms if result else []

    def find_common_scaffolds(self,
                              smiles_list: List[str],
                              patterns: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:
        """
        Find which molecules match common scaffold patterns.

        Args:
            smiles_list: List of SMILES strings
            patterns: Optional custom patterns dict, uses CommonPatterns if None

        Returns:
            Dict mapping pattern names to lists of matching SMILES
        """
        if not self._is_ready:
            return {}

        if patterns is None:
            patterns = CommonPatterns.get_quat_patterns()

        results = {name: [] for name in patterns}

        for smiles in smiles_list:
            for name, smarts in patterns.items():
                if self.has_substructure(smiles, smarts):
                    results[name].append(smiles)

        return results

    def filter_by_substructure(self,
                               smiles_list: List[str],
                               required_patterns: List[str] = None,
                               excluded_patterns: List[str] = None) -> List[str]:
        """
        Filter molecules by required and excluded substructures.

        Args:
            smiles_list: List of SMILES strings
            required_patterns: List of SMARTS that must be present
            excluded_patterns: List of SMARTS that must be absent

        Returns:
            Filtered list of SMILES
        """
        if not self._is_ready:
            return smiles_list

        required_patterns = required_patterns or []
        excluded_patterns = excluded_patterns or []

        filtered = []
        for smiles in smiles_list:
            # Check required patterns
            has_all_required = all(
                self.has_substructure(smiles, p) for p in required_patterns
            )
            if not has_all_required:
                continue

            # Check excluded patterns
            has_any_excluded = any(
                self.has_substructure(smiles, p) for p in excluded_patterns
            )
            if has_any_excluded:
                continue

            filtered.append(smiles)

        return filtered

    def classify_quat_type(self, smiles: str) -> Optional[str]:
        """
        Classify the type of quaternary ammonium compound.

        Returns one of:
        - 'benzalkonium': Benzalkonium-like (BAC)
        - 'didecyl': Didecyl quaternary (DDAC)
        - 'cetyl': Cetyl quaternary (CPC, CTAB)
        - 'pyridinium': Pyridinium-based
        - 'imidazolium': Imidazolium-based
        - 'other_quat': Other quaternary type
        - None: Not a quaternary ammonium
        """
        if not self._is_ready:
            return None

        if not self.has_quat_nitrogen(smiles):
            return None

        # Check specific types
        if self.has_substructure(smiles, CommonPatterns.BAC_CORE):
            return "benzalkonium"
        if self.has_substructure(smiles, CommonPatterns.DDAC_CORE):
            return "didecyl"
        if self.has_substructure(smiles, CommonPatterns.CETYL_CORE):
            return "cetyl"
        if self.has_substructure(smiles, CommonPatterns.PYRIDINIUM):
            return "pyridinium"
        if self.has_substructure(smiles, CommonPatterns.IMIDAZOLIUM):
            return "imidazolium"

        return "other_quat"

    def get_structural_features(self, smiles: str) -> Dict[str, Any]:
        """
        Get structural feature summary for a molecule.

        Args:
            smiles: SMILES string

        Returns:
            Dict of structural features
        """
        if not self._is_ready:
            return {}

        features = {
            "has_quat_nitrogen": self.has_quat_nitrogen(smiles),
            "quat_type": self.classify_quat_type(smiles),
            "has_benzyl": self.has_substructure(smiles, CommonPatterns.BENZYL),
            "has_phenyl": self.has_substructure(smiles, CommonPatterns.PHENYL),
            "has_long_chain": self.has_substructure(smiles, CommonPatterns.LONG_ALKYL_CHAIN),
            "has_hydroxyl": self.has_substructure(smiles, CommonPatterns.HYDROXYL),
            "has_ether": self.has_substructure(smiles, CommonPatterns.ETHER),
            "has_ester": self.has_substructure(smiles, CommonPatterns.ESTER),
            "has_amide": self.has_substructure(smiles, CommonPatterns.AMIDE),
            "has_chloro": self.has_substructure(smiles, CommonPatterns.CHLORO),
            "has_bromo": self.has_substructure(smiles, CommonPatterns.BROMO),
            "has_fluoro": self.has_substructure(smiles, CommonPatterns.FLUORO),
        }

        return features

    def similarity_search(self,
                          query_smiles: str,
                          molecules: List[Dict[str, Any]],
                          threshold: float = 0.7,
                          max_results: int = 100) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find molecules similar to query using Tanimoto similarity.

        Args:
            query_smiles: Query SMILES
            molecules: List of molecule dicts with 'smiles' key
            threshold: Minimum similarity threshold (0-1)
            max_results: Maximum results to return

        Returns:
            List of (molecule_dict, similarity) tuples sorted by similarity
        """
        if not self._is_ready:
            return []

        try:
            query_mol = Chem.MolFromSmiles(query_smiles)
            if query_mol is None:
                return []

            query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)

            results = []
            for mol_data in molecules:
                smiles = mol_data.get("smiles")
                if not smiles:
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                similarity = DataStructs.TanimotoSimilarity(query_fp, fp)

                if similarity >= threshold:
                    results.append((mol_data, similarity))

            # Sort by similarity descending
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:max_results]

        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return []

    def clear_cache(self):
        """Clear the pattern cache"""
        self._pattern_cache.clear()
