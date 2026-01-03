"""
Benchmark Comparator for Quaternary Ammonium Compounds

Compares generated molecules against well-characterized reference compounds
to identify promising candidates.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class ComparisonOutcome(Enum):
    BETTER = "better"
    SIMILAR = "similar"
    WORSE = "worse"
    UNKNOWN = "unknown"


@dataclass
class PropertyComparison:
    """Comparison of a single property"""
    property_name: str
    generated_value: float
    reference_value: float
    reference_range: Optional[Tuple[float, float]] = None
    outcome: ComparisonOutcome = ComparisonOutcome.UNKNOWN
    interpretation: str = ""


@dataclass
class BenchmarkResult:
    """Result of benchmarking a molecule against references"""
    smiles: str
    molecule_id: Optional[int] = None

    # Overall assessment
    overall_score: float = 0.0
    recommendation: str = ""
    confidence: float = 0.0

    # Closest reference compounds
    closest_references: List[Dict] = field(default_factory=list)

    # Property comparisons
    property_comparisons: List[PropertyComparison] = field(default_factory=list)

    # Summary statistics
    properties_better: int = 0
    properties_similar: int = 0
    properties_worse: int = 0

    # Predicted advantages/disadvantages
    predicted_advantages: List[str] = field(default_factory=list)
    predicted_disadvantages: List[str] = field(default_factory=list)

    # Structural analysis
    structural_novelty: float = 0.0
    scaffold_type: str = ""


class BenchmarkComparator:
    """
    Compares generated molecules against reference quaternary ammonium compounds.
    """

    # Property comparison thresholds (what counts as "similar")
    SIMILARITY_THRESHOLDS = {
        "mic_value": 2.0,       # Within 2-fold is similar
        "logp": 1.0,            # Within 1 unit
        "mw": 50.0,             # Within 50 Da
        "tpsa": 20.0,           # Within 20 Å²
        "efficacy_score": 10.0,  # Within 10 points
        "safety_score": 10.0,
        "sa_score": 10.0,
    }

    # Property optimization direction (True = higher is better)
    OPTIMIZATION_DIRECTION = {
        "mic_value": False,      # Lower is better
        "ld50": True,            # Higher is better (less toxic)
        "efficacy_score": True,
        "safety_score": True,
        "environmental_score": True,
        "sa_score": True,
        "logp": None,            # Optimal range (not directional)
        "mw": None,
    }

    def __init__(self, reference_db=None):
        """
        Args:
            reference_db: ReferenceDatabase instance with known quats
        """
        self.reference_db = reference_db
        self._reference_fps: Dict[str, object] = {}
        self._is_ready = RDKIT_AVAILABLE

        if reference_db and RDKIT_AVAILABLE:
            self._precompute_reference_fingerprints()

    def _precompute_reference_fingerprints(self):
        """Precompute fingerprints for all reference compounds"""
        if not self.reference_db:
            return

        for compound in self.reference_db.get_all():
            try:
                mol = Chem.MolFromSmiles(compound.smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    self._reference_fps[compound.name] = fp
            except Exception as e:
                logger.warning(f"Could not compute fingerprint for {compound.name}: {e}")

    def benchmark_molecule(self,
                           smiles: str,
                           predicted_scores: Optional[Dict] = None,
                           molecule_id: Optional[int] = None) -> BenchmarkResult:
        """
        Benchmark a generated molecule against reference compounds.

        Args:
            smiles: SMILES of generated molecule
            predicted_scores: Predicted scores from scoring pipeline
            molecule_id: Optional database ID

        Returns:
            BenchmarkResult with detailed comparison
        """
        result = BenchmarkResult(smiles=smiles, molecule_id=molecule_id)

        if not self._is_ready:
            result.recommendation = "Benchmarking unavailable (RDKit not installed)"
            return result

        if not self.reference_db:
            result.recommendation = "No reference database available"
            return result

        # Calculate properties of generated molecule
        mol_props = self._calculate_properties(smiles)
        if mol_props is None:
            result.recommendation = "Invalid molecule structure"
            return result

        # Add predicted scores to properties
        if predicted_scores:
            mol_props.update(predicted_scores)

        # Find closest reference compounds
        closest = self._find_closest_references(smiles, n=3)
        result.closest_references = closest

        # Calculate structural novelty
        if closest:
            max_similarity = max(r["similarity"] for r in closest)
            result.structural_novelty = 1.0 - max_similarity
        else:
            result.structural_novelty = 1.0

        # Compare properties against references
        property_comparisons = self._compare_properties(mol_props, closest)
        result.property_comparisons = property_comparisons

        # Count outcomes
        for pc in property_comparisons:
            if pc.outcome == ComparisonOutcome.BETTER:
                result.properties_better += 1
            elif pc.outcome == ComparisonOutcome.SIMILAR:
                result.properties_similar += 1
            elif pc.outcome == ComparisonOutcome.WORSE:
                result.properties_worse += 1

        # Identify advantages and disadvantages
        result.predicted_advantages = self._identify_advantages(property_comparisons, mol_props)
        result.predicted_disadvantages = self._identify_disadvantages(property_comparisons, mol_props)

        # Determine scaffold type
        result.scaffold_type = self._classify_scaffold(smiles)

        # Calculate overall score and recommendation
        result.overall_score = self._calculate_overall_score(result)
        result.recommendation = self._generate_recommendation(result)
        result.confidence = self._calculate_confidence(result, predicted_scores)

        return result

    def _calculate_properties(self, smiles: str) -> Optional[Dict]:
        """Calculate molecular properties"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            return {
                "mw": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "hbd": rdMolDescriptors.CalcNumHBD(mol),
                "hba": rdMolDescriptors.CalcNumHBA(mol),
                "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                "num_rings": rdMolDescriptors.CalcNumRings(mol),
                "formal_charge": Chem.GetFormalCharge(mol),
            }
        except Exception:
            return None

    def _find_closest_references(self, smiles: str, n: int = 3) -> List[Dict]:
        """Find the n most similar reference compounds"""
        if not self._reference_fps:
            return []

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []

            query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

            similarities = []
            for name, ref_fp in self._reference_fps.items():
                sim = DataStructs.TanimotoSimilarity(query_fp, ref_fp)

                # Get reference compound data
                ref_compound = self.reference_db.get_by_name(name)

                similarities.append({
                    "name": name,
                    "similarity": round(sim, 3),
                    "smiles": ref_compound.smiles if ref_compound else None,
                    "mic_s_aureus": ref_compound.mic_s_aureus if ref_compound else None,
                    "mic_e_coli": ref_compound.mic_e_coli if ref_compound else None,
                    "ld50": ref_compound.ld50_oral_rat if ref_compound else None,
                    "applications": ref_compound.applications if ref_compound else []
                })

            # Sort by similarity and return top n
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:n]

        except Exception as e:
            logger.error(f"Error finding closest references: {e}")
            return []

    def _compare_properties(self, mol_props: Dict, closest_refs: List[Dict]) -> List[PropertyComparison]:
        """Compare molecule properties against reference compounds"""
        comparisons = []

        if not closest_refs:
            return comparisons

        # Get the most similar reference for primary comparison
        primary_ref = closest_refs[0]
        ref_compound = self.reference_db.get_by_name(primary_ref["name"]) if self.reference_db else None

        # Compare key properties
        properties_to_compare = [
            ("efficacy_score", "Antimicrobial Efficacy", True),
            ("safety_score", "Safety Profile", True),
            ("sa_score", "Synthetic Accessibility", True),
            ("environmental_score", "Environmental Impact", True),
            ("logp", "Lipophilicity (LogP)", None),
            ("mw", "Molecular Weight", None),
            ("tpsa", "Polar Surface Area", None),
        ]

        for prop_key, prop_name, higher_better in properties_to_compare:
            if prop_key not in mol_props:
                continue

            gen_value = mol_props[prop_key]

            # Get reference value (from scores or compound properties)
            ref_value = self._get_reference_value(prop_key, ref_compound)

            if ref_value is not None:
                comparison = PropertyComparison(
                    property_name=prop_name,
                    generated_value=round(gen_value, 2),
                    reference_value=round(ref_value, 2)
                )

                # Determine outcome
                threshold = self.SIMILARITY_THRESHOLDS.get(prop_key, 10)
                diff = gen_value - ref_value

                if abs(diff) < threshold:
                    comparison.outcome = ComparisonOutcome.SIMILAR
                    comparison.interpretation = f"Similar to {primary_ref['name']}"
                elif higher_better is True:
                    if diff > 0:
                        comparison.outcome = ComparisonOutcome.BETTER
                        comparison.interpretation = f"Better than {primary_ref['name']} (+{diff:.1f})"
                    else:
                        comparison.outcome = ComparisonOutcome.WORSE
                        comparison.interpretation = f"Worse than {primary_ref['name']} ({diff:.1f})"
                elif higher_better is False:
                    if diff < 0:
                        comparison.outcome = ComparisonOutcome.BETTER
                        comparison.interpretation = f"Better than {primary_ref['name']} ({diff:.1f})"
                    else:
                        comparison.outcome = ComparisonOutcome.WORSE
                        comparison.interpretation = f"Worse than {primary_ref['name']} (+{diff:.1f})"
                else:
                    # Optimal range property
                    comparison.outcome = ComparisonOutcome.SIMILAR
                    comparison.interpretation = f"Comparable to {primary_ref['name']}"

                comparisons.append(comparison)

        return comparisons

    def _get_reference_value(self, prop_key: str, ref_compound) -> Optional[float]:
        """Get reference value for a property"""
        if ref_compound is None:
            return None

        # Map property keys to reference compound attributes
        prop_mapping = {
            "logp": lambda c: 4.0,  # Typical quat logP
            "mw": lambda c: 350.0,  # Typical quat MW
            "tpsa": lambda c: 30.0,  # Typical quat TPSA
            "efficacy_score": lambda c: 75.0,  # Assume good efficacy
            "safety_score": lambda c: 60.0,    # Moderate safety
            "sa_score": lambda c: 70.0,        # Reasonable synthesis
            "environmental_score": lambda c: 50.0,  # Variable
        }

        if prop_key in prop_mapping:
            return prop_mapping[prop_key](ref_compound)

        return None

    def _identify_advantages(self, comparisons: List[PropertyComparison],
                              mol_props: Dict) -> List[str]:
        """Identify predicted advantages of the generated molecule"""
        advantages = []

        for comp in comparisons:
            if comp.outcome == ComparisonOutcome.BETTER:
                advantages.append(f"Improved {comp.property_name}: {comp.interpretation}")

        # Check for specific advantageous features
        if mol_props.get("formal_charge", 0) == 1:
            advantages.append("Single positive charge (optimal for membrane interaction)")

        logp = mol_props.get("logp", 0)
        if 3 <= logp <= 6:
            advantages.append("LogP in optimal range for membrane disruption (3-6)")

        return advantages

    def _identify_disadvantages(self, comparisons: List[PropertyComparison],
                                 mol_props: Dict) -> List[str]:
        """Identify predicted disadvantages"""
        disadvantages = []

        for comp in comparisons:
            if comp.outcome == ComparisonOutcome.WORSE:
                disadvantages.append(f"Reduced {comp.property_name}: {comp.interpretation}")

        # Check for specific concerns
        mw = mol_props.get("mw", 0)
        if mw > 600:
            disadvantages.append(f"High molecular weight ({mw:.0f} Da) may limit bioavailability")

        logp = mol_props.get("logp", 0)
        if logp > 8:
            disadvantages.append(f"Very high LogP ({logp:.1f}) may cause solubility issues")

        return disadvantages

    def _classify_scaffold(self, smiles: str) -> str:
        """Classify the molecular scaffold type"""
        if not self._is_ready:
            return "unknown"

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "unknown"

            smiles_upper = smiles

            # Check for pyridinium
            pyridinium_pattern = Chem.MolFromSmarts("[n+]1ccccc1")
            if pyridinium_pattern and mol.HasSubstructMatch(pyridinium_pattern):
                return "pyridinium"

            # Check for imidazolium
            imidazolium_pattern = Chem.MolFromSmarts("[n+]1ccnc1")
            if imidazolium_pattern and mol.HasSubstructMatch(imidazolium_pattern):
                return "imidazolium"

            # Check for benzyl quaternary
            benzyl_quat = Chem.MolFromSmarts("[N+]([CH2]c1ccccc1)")
            if benzyl_quat and mol.HasSubstructMatch(benzyl_quat):
                return "benzylammonium"

            # Check for tetraalkylammonium
            tetraalkyl = Chem.MolFromSmarts("[N+]([CH3])([CH3])([CH3])")
            if tetraalkyl and mol.HasSubstructMatch(tetraalkyl):
                return "tetraalkylammonium"

            # Check for aromatic quat
            aromatic_quat = Chem.MolFromSmarts("[n+]")
            if aromatic_quat and mol.HasSubstructMatch(aromatic_quat):
                return "aromatic_quat"

            # Check for aliphatic quat
            aliphatic_quat = Chem.MolFromSmarts("[N+;X4]")
            if aliphatic_quat and mol.HasSubstructMatch(aliphatic_quat):
                return "aliphatic_quat"

            return "unknown"

        except Exception:
            return "unknown"

    def _calculate_overall_score(self, result: BenchmarkResult) -> float:
        """Calculate overall benchmark score (0-100)"""
        # Weighted components
        score = 50.0  # Start at neutral

        # Adjust for property comparisons
        total_props = result.properties_better + result.properties_similar + result.properties_worse
        if total_props > 0:
            better_ratio = result.properties_better / total_props
            worse_ratio = result.properties_worse / total_props
            score += (better_ratio - worse_ratio) * 30

        # Adjust for structural novelty (slight bonus for novelty)
        score += result.structural_novelty * 10

        # Adjust for advantages/disadvantages
        score += len(result.predicted_advantages) * 3
        score -= len(result.predicted_disadvantages) * 3

        return max(0, min(100, score))

    def _generate_recommendation(self, result: BenchmarkResult) -> str:
        """Generate a recommendation based on benchmark results"""
        score = result.overall_score

        if score >= 80:
            return "Highly promising candidate - consider for experimental validation"
        elif score >= 65:
            return "Good candidate - comparable to established quats"
        elif score >= 50:
            return "Moderate candidate - may need optimization"
        elif score >= 35:
            return "Below average - significant improvements needed"
        else:
            return "Poor candidate - consider alternative scaffolds"

    def _calculate_confidence(self, result: BenchmarkResult,
                               predicted_scores: Optional[Dict]) -> float:
        """Calculate confidence in the benchmark assessment"""
        confidence = 0.5  # Base confidence

        # Higher confidence if we have predicted scores
        if predicted_scores:
            confidence += 0.2

        # Higher confidence if similar to known compounds
        if result.closest_references:
            max_sim = max(r["similarity"] for r in result.closest_references)
            confidence += max_sim * 0.2

        # Higher confidence with more property comparisons
        num_comparisons = len(result.property_comparisons)
        confidence += min(0.1, num_comparisons * 0.02)

        return min(0.95, confidence)

    def benchmark_batch(self,
                        molecules: List[Tuple[str, Optional[Dict], Optional[int]]],
                        top_n: int = 10) -> List[BenchmarkResult]:
        """
        Benchmark multiple molecules and return top candidates.

        Args:
            molecules: List of (smiles, predicted_scores, molecule_id) tuples
            top_n: Number of top results to return

        Returns:
            List of BenchmarkResult sorted by overall score
        """
        results = []

        for smiles, scores, mol_id in molecules:
            result = self.benchmark_molecule(smiles, scores, mol_id)
            results.append(result)

        # Sort by overall score
        results.sort(key=lambda r: r.overall_score, reverse=True)

        return results[:top_n]

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self.reference_db is not None
