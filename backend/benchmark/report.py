"""
Benchmark Report Generator

Generates formatted reports comparing generated molecules to references.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass

from benchmark.comparator import BenchmarkResult, ComparisonOutcome

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkReport:
    """Complete benchmark report for a generation run"""
    generated_at: str
    total_molecules: int
    molecules_benchmarked: int

    # Summary statistics
    avg_overall_score: float
    top_candidates_count: int
    scaffold_distribution: Dict[str, int]

    # Top molecules
    top_candidates: List[Dict]

    # Comparison to references
    reference_comparison_summary: Dict

    # Recommendations
    recommendations: List[str]

    @classmethod
    def generate(cls,
                 results: List[BenchmarkResult],
                 reference_db=None) -> "BenchmarkReport":
        """Generate a benchmark report from results"""

        if not results:
            return cls(
                generated_at=datetime.now().isoformat(),
                total_molecules=0,
                molecules_benchmarked=0,
                avg_overall_score=0,
                top_candidates_count=0,
                scaffold_distribution={},
                top_candidates=[],
                reference_comparison_summary={},
                recommendations=["No molecules to benchmark"]
            )

        # Calculate statistics
        scores = [r.overall_score for r in results]
        avg_score = sum(scores) / len(scores)

        # Count scaffolds
        scaffold_dist = {}
        for r in results:
            scaffold = r.scaffold_type or "unknown"
            scaffold_dist[scaffold] = scaffold_dist.get(scaffold, 0) + 1

        # Identify top candidates (score >= 65)
        top_results = [r for r in results if r.overall_score >= 65]
        top_candidates = [
            {
                "smiles": r.smiles,
                "molecule_id": r.molecule_id,
                "overall_score": r.overall_score,
                "recommendation": r.recommendation,
                "scaffold_type": r.scaffold_type,
                "structural_novelty": r.structural_novelty,
                "advantages": r.predicted_advantages,
                "closest_reference": r.closest_references[0]["name"] if r.closest_references else None
            }
            for r in sorted(top_results, key=lambda x: x.overall_score, reverse=True)[:20]
        ]

        # Reference comparison summary
        ref_summary = cls._generate_reference_summary(results, reference_db)

        # Generate recommendations
        recommendations = cls._generate_recommendations(results, scaffold_dist, avg_score)

        return cls(
            generated_at=datetime.now().isoformat(),
            total_molecules=len(results),
            molecules_benchmarked=len(results),
            avg_overall_score=round(avg_score, 1),
            top_candidates_count=len(top_candidates),
            scaffold_distribution=scaffold_dist,
            top_candidates=top_candidates,
            reference_comparison_summary=ref_summary,
            recommendations=recommendations
        )

    @staticmethod
    def _generate_reference_summary(results: List[BenchmarkResult],
                                     reference_db) -> Dict:
        """Generate summary of comparisons to reference compounds"""
        summary = {
            "total_comparisons": 0,
            "better_than_reference": 0,
            "similar_to_reference": 0,
            "worse_than_reference": 0,
            "closest_references_used": {},
            "property_improvements": {},
            "property_deficits": {}
        }

        for result in results:
            # Count property outcomes
            for comp in result.property_comparisons:
                summary["total_comparisons"] += 1
                if comp.outcome == ComparisonOutcome.BETTER:
                    summary["better_than_reference"] += 1
                    prop = comp.property_name
                    summary["property_improvements"][prop] = \
                        summary["property_improvements"].get(prop, 0) + 1
                elif comp.outcome == ComparisonOutcome.SIMILAR:
                    summary["similar_to_reference"] += 1
                elif comp.outcome == ComparisonOutcome.WORSE:
                    summary["worse_than_reference"] += 1
                    prop = comp.property_name
                    summary["property_deficits"][prop] = \
                        summary["property_deficits"].get(prop, 0) + 1

            # Track which references are used
            for ref in result.closest_references:
                name = ref["name"]
                summary["closest_references_used"][name] = \
                    summary["closest_references_used"].get(name, 0) + 1

        return summary

    @staticmethod
    def _generate_recommendations(results: List[BenchmarkResult],
                                   scaffold_dist: Dict,
                                   avg_score: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Overall performance
        if avg_score >= 65:
            recommendations.append(
                f"Generation quality is good (avg score: {avg_score:.1f}). "
                "Top candidates are ready for experimental validation."
            )
        elif avg_score >= 50:
            recommendations.append(
                f"Generation quality is moderate (avg score: {avg_score:.1f}). "
                "Consider adjusting optimization weights or constraints."
            )
        else:
            recommendations.append(
                f"Generation quality needs improvement (avg score: {avg_score:.1f}). "
                "Review generation parameters and scoring weights."
            )

        # Scaffold diversity
        if len(scaffold_dist) == 1:
            recommendations.append(
                f"All molecules have same scaffold ({list(scaffold_dist.keys())[0]}). "
                "Consider diversifying structural exploration."
            )
        elif len(scaffold_dist) >= 4:
            recommendations.append(
                "Good scaffold diversity achieved. "
                "Multiple structural classes represented."
            )

        # Top candidates
        promising = [r for r in results if r.overall_score >= 70]
        if promising:
            recommendations.append(
                f"{len(promising)} highly promising candidates identified. "
                "Prioritize these for synthesis planning."
            )

        # Common improvements
        improvements = {}
        for r in results:
            for adv in r.predicted_advantages:
                key = adv.split(":")[0] if ":" in adv else adv
                improvements[key] = improvements.get(key, 0) + 1

        if improvements:
            top_improvement = max(improvements.items(), key=lambda x: x[1])
            recommendations.append(
                f"Most common improvement: {top_improvement[0]} "
                f"({top_improvement[1]} molecules)"
            )

        return recommendations

    def to_dict(self) -> Dict:
        """Convert report to dictionary"""
        return {
            "generated_at": self.generated_at,
            "summary": {
                "total_molecules": self.total_molecules,
                "molecules_benchmarked": self.molecules_benchmarked,
                "avg_overall_score": self.avg_overall_score,
                "top_candidates_count": self.top_candidates_count
            },
            "scaffold_distribution": self.scaffold_distribution,
            "top_candidates": self.top_candidates,
            "reference_comparison": self.reference_comparison_summary,
            "recommendations": self.recommendations
        }
