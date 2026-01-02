"""
Reference database of known quaternary ammonium compounds

Provides well-characterized benchmark compounds with literature MIC values
for validation and comparison of generated molecules.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ReferenceQuat:
    """A well-characterized reference quaternary ammonium compound"""

    name: str
    smiles: str
    cas_number: Optional[str] = None
    chembl_id: Optional[str] = None

    # Typical MIC ranges (ug/mL) from literature - (min, max)
    mic_s_aureus: Optional[Tuple[float, float]] = None
    mic_e_coli: Optional[Tuple[float, float]] = None
    mic_p_aeruginosa: Optional[Tuple[float, float]] = None
    mic_c_albicans: Optional[Tuple[float, float]] = None

    # Safety data
    ld50_oral_rat: Optional[float] = None   # mg/kg
    skin_irritation: Optional[str] = None   # "mild", "moderate", "severe"
    eye_irritation: Optional[str] = None

    # Environmental
    biodegradability: Optional[str] = None  # "readily", "inherent", "not"
    aquatic_toxicity_fish_lc50: Optional[float] = None  # mg/L

    # Applications
    applications: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "smiles": self.smiles,
            "cas_number": self.cas_number,
            "chembl_id": self.chembl_id,
            "mic_s_aureus": self.mic_s_aureus,
            "mic_e_coli": self.mic_e_coli,
            "mic_p_aeruginosa": self.mic_p_aeruginosa,
            "mic_c_albicans": self.mic_c_albicans,
            "ld50_oral_rat": self.ld50_oral_rat,
            "skin_irritation": self.skin_irritation,
            "eye_irritation": self.eye_irritation,
            "biodegradability": self.biodegradability,
            "aquatic_toxicity_fish_lc50": self.aquatic_toxicity_fish_lc50,
            "applications": self.applications
        }


# Curated database of well-known quaternary ammonium disinfectants
# MIC values are from published literature reviews
REFERENCE_QUATS = [
    ReferenceQuat(
        name="Benzalkonium chloride (C12)",
        smiles="CCCCCCCCCCCC[N+](C)(C)Cc1ccccc1.[Cl-]",
        cas_number="8001-54-5",
        chembl_id="CHEMBL578",
        mic_s_aureus=(1, 8),
        mic_e_coli=(8, 64),
        mic_p_aeruginosa=(32, 256),
        mic_c_albicans=(4, 16),
        ld50_oral_rat=240,
        skin_irritation="moderate",
        eye_irritation="severe",
        biodegradability="inherent",
        aquatic_toxicity_fish_lc50=0.5,
        applications=["surface disinfectant", "antiseptic", "preservative"]
    ),
    ReferenceQuat(
        name="Cetylpyridinium chloride",
        smiles="CCCCCCCCCCCCCCCC[n+]1ccccc1.[Cl-]",
        cas_number="123-03-5",
        chembl_id="CHEMBL1354",
        mic_s_aureus=(0.5, 4),
        mic_e_coli=(4, 32),
        mic_p_aeruginosa=(16, 128),
        mic_c_albicans=(2, 8),
        ld50_oral_rat=200,
        skin_irritation="mild",
        eye_irritation="moderate",
        biodegradability="inherent",
        aquatic_toxicity_fish_lc50=0.3,
        applications=["mouthwash", "throat lozenges", "surface disinfectant"]
    ),
    ReferenceQuat(
        name="Didecyldimethylammonium chloride (DDAC)",
        smiles="CCCCCCCCCC[N+](C)(C)CCCCCCCCCC.[Cl-]",
        cas_number="7173-51-5",
        chembl_id="CHEMBL1201135",
        mic_s_aureus=(0.5, 4),
        mic_e_coli=(2, 16),
        mic_p_aeruginosa=(8, 64),
        mic_c_albicans=(1, 8),
        ld50_oral_rat=84,
        skin_irritation="moderate",
        eye_irritation="severe",
        biodegradability="readily",
        aquatic_toxicity_fish_lc50=0.2,
        applications=["hard surface disinfectant", "algaecide", "wood preservative"]
    ),
    ReferenceQuat(
        name="Benzethonium chloride",
        smiles="CC(C)(C)CC(C)(C)c1ccc(OCCOCC[N+](C)(C)Cc2ccccc2)cc1.[Cl-]",
        cas_number="121-54-0",
        chembl_id="CHEMBL1236088",
        mic_s_aureus=(1, 8),
        mic_e_coli=(8, 32),
        mic_p_aeruginosa=(32, 128),
        mic_c_albicans=(4, 16),
        ld50_oral_rat=368,
        skin_irritation="mild",
        eye_irritation="moderate",
        biodegradability="inherent",
        aquatic_toxicity_fish_lc50=1.0,
        applications=["antiseptic", "cosmetic preservative", "first aid"]
    ),
    ReferenceQuat(
        name="Cetrimonium bromide (CTAB)",
        smiles="CCCCCCCCCCCCCCCC[N+](C)(C)C.[Br-]",
        cas_number="57-09-0",
        chembl_id="CHEMBL447964",
        mic_s_aureus=(1, 4),
        mic_e_coli=(4, 16),
        mic_p_aeruginosa=(16, 64),
        mic_c_albicans=(2, 8),
        ld50_oral_rat=410,
        skin_irritation="moderate",
        eye_irritation="severe",
        biodegradability="inherent",
        aquatic_toxicity_fish_lc50=0.4,
        applications=["hair conditioner", "antiseptic", "DNA extraction"]
    ),
    ReferenceQuat(
        name="Octenidine dihydrochloride",
        smiles="CCCCCCCCCCN=C(N)N=C(N)c1ccc(C(C)(C)C)cc1.CCCCCCCCCCN=C(N)N=C(N)c1ccc(C(C)(C)C)cc1.[Cl-].[Cl-]",
        cas_number="70775-75-6",
        chembl_id=None,
        mic_s_aureus=(0.25, 2),
        mic_e_coli=(0.5, 4),
        mic_p_aeruginosa=(2, 16),
        mic_c_albicans=(0.5, 4),
        ld50_oral_rat=1850,
        skin_irritation="mild",
        eye_irritation="mild",
        biodegradability="inherent",
        aquatic_toxicity_fish_lc50=None,
        applications=["wound antiseptic", "mucous membrane disinfection"]
    ),
    ReferenceQuat(
        name="Domiphen bromide",
        smiles="CCCCCCCCCCCC[N+](C)(C)CCOc1ccccc1.[Br-]",
        cas_number="538-71-6",
        chembl_id="CHEMBL1201247",
        mic_s_aureus=(1, 8),
        mic_e_coli=(8, 32),
        mic_p_aeruginosa=(32, 128),
        mic_c_albicans=(4, 16),
        ld50_oral_rat=320,
        skin_irritation="mild",
        eye_irritation="moderate",
        biodegradability="inherent",
        aquatic_toxicity_fish_lc50=0.8,
        applications=["throat lozenges", "surface disinfectant", "antiseptic"]
    ),
    ReferenceQuat(
        name="Dequalinium chloride",
        smiles="Cc1cc2c(N)cc(N)cc2[n+](CCCCCCCCCC[n+]3c4cc(N)cc(N)c4cc3C)c1.[Cl-].[Cl-]",
        cas_number="522-51-0",
        chembl_id="CHEMBL1523",
        mic_s_aureus=(0.5, 4),
        mic_e_coli=(4, 16),
        mic_p_aeruginosa=(8, 64),
        mic_c_albicans=(0.5, 4),
        ld50_oral_rat=150,
        skin_irritation="mild",
        eye_irritation="mild",
        biodegradability="inherent",
        aquatic_toxicity_fish_lc50=0.5,
        applications=["throat lozenges", "vaginal antiseptic", "oral antiseptic"]
    ),
]


class ReferenceDatabase:
    """
    Database of reference quaternary ammonium compounds for validation.

    Provides benchmark data and comparison tools for evaluating
    generated molecules against established disinfectants.
    """

    def __init__(self):
        """Initialize the reference database"""
        self.compounds = {c.name: c for c in REFERENCE_QUATS}
        self.by_smiles = {c.smiles: c for c in REFERENCE_QUATS}
        self.by_chembl = {c.chembl_id: c for c in REFERENCE_QUATS if c.chembl_id}
        self.by_cas = {c.cas_number: c for c in REFERENCE_QUATS if c.cas_number}

    def get_all(self) -> List[ReferenceQuat]:
        """Get all reference compounds"""
        return list(self.compounds.values())

    def get_by_name(self, name: str) -> Optional[ReferenceQuat]:
        """Get compound by name (case-insensitive partial match)"""
        name_lower = name.lower()
        for compound in self.compounds.values():
            if name_lower in compound.name.lower():
                return compound
        return None

    def get_by_smiles(self, smiles: str) -> Optional[ReferenceQuat]:
        """Get compound by exact SMILES match"""
        return self.by_smiles.get(smiles)

    def get_by_chembl_id(self, chembl_id: str) -> Optional[ReferenceQuat]:
        """Get compound by ChEMBL ID"""
        return self.by_chembl.get(chembl_id)

    def get_by_cas(self, cas_number: str) -> Optional[ReferenceQuat]:
        """Get compound by CAS number"""
        return self.by_cas.get(cas_number)

    def get_benchmarks(self) -> Dict:
        """
        Get benchmark data for evaluating generated molecules.

        Returns categorized MIC thresholds and reference compounds.
        """
        benchmarks = {
            "gram_positive": {
                "target": "Staphylococcus aureus",
                "excellent_mic": 2,   # ug/mL
                "good_mic": 8,
                "moderate_mic": 32,
                "reference_compounds": []
            },
            "gram_negative": {
                "target": "Escherichia coli",
                "excellent_mic": 4,
                "good_mic": 16,
                "moderate_mic": 64,
                "reference_compounds": []
            },
            "pseudomonas": {
                "target": "Pseudomonas aeruginosa",
                "excellent_mic": 16,
                "good_mic": 64,
                "moderate_mic": 128,
                "reference_compounds": []
            },
            "antifungal": {
                "target": "Candida albicans",
                "excellent_mic": 2,
                "good_mic": 8,
                "moderate_mic": 32,
                "reference_compounds": []
            }
        }

        for compound in self.compounds.values():
            if compound.mic_s_aureus:
                benchmarks["gram_positive"]["reference_compounds"].append({
                    "name": compound.name,
                    "mic_range": compound.mic_s_aureus
                })
            if compound.mic_e_coli:
                benchmarks["gram_negative"]["reference_compounds"].append({
                    "name": compound.name,
                    "mic_range": compound.mic_e_coli
                })
            if compound.mic_p_aeruginosa:
                benchmarks["pseudomonas"]["reference_compounds"].append({
                    "name": compound.name,
                    "mic_range": compound.mic_p_aeruginosa
                })
            if compound.mic_c_albicans:
                benchmarks["antifungal"]["reference_compounds"].append({
                    "name": compound.name,
                    "mic_range": compound.mic_c_albicans
                })

        return benchmarks

    def get_safety_benchmarks(self) -> Dict:
        """Get safety benchmark data from reference compounds"""
        benchmarks = {
            "ld50_range": {
                "safest": None,
                "most_toxic": None,
                "values": []
            },
            "skin_irritation": {
                "mild": [],
                "moderate": [],
                "severe": []
            },
            "eye_irritation": {
                "mild": [],
                "moderate": [],
                "severe": []
            }
        }

        for compound in self.compounds.values():
            if compound.ld50_oral_rat:
                benchmarks["ld50_range"]["values"].append({
                    "name": compound.name,
                    "ld50": compound.ld50_oral_rat
                })

            if compound.skin_irritation:
                benchmarks["skin_irritation"].get(compound.skin_irritation, []).append(compound.name)

            if compound.eye_irritation:
                benchmarks["eye_irritation"].get(compound.eye_irritation, []).append(compound.name)

        # Find safest/most toxic
        if benchmarks["ld50_range"]["values"]:
            sorted_ld50 = sorted(benchmarks["ld50_range"]["values"], key=lambda x: x["ld50"])
            benchmarks["ld50_range"]["most_toxic"] = sorted_ld50[0]
            benchmarks["ld50_range"]["safest"] = sorted_ld50[-1]

        return benchmarks

    def compare_to_references(self, smiles: str, predicted_scores: Dict) -> Dict:
        """
        Compare a generated molecule's predicted scores to reference compounds.

        Args:
            smiles: SMILES of the generated molecule
            predicted_scores: Dictionary with efficacy, safety, environmental scores

        Returns:
            Comparison results with recommendations
        """
        comparison = {
            "similar_references": [],
            "better_than": [],
            "worse_than": [],
            "recommendation": "",
            "percentile_estimate": None
        }

        efficacy = predicted_scores.get("efficacy", 50)
        safety = predicted_scores.get("safety", 50)
        environmental = predicted_scores.get("environmental", 50)

        # Calculate combined score
        combined = (efficacy + safety + environmental) / 3

        # Estimate percentile among references (rough)
        if combined > 75:
            comparison["percentile_estimate"] = 90
        elif combined > 65:
            comparison["percentile_estimate"] = 70
        elif combined > 55:
            comparison["percentile_estimate"] = 50
        else:
            comparison["percentile_estimate"] = 30

        # Generate recommendation
        if efficacy > 70 and safety > 60 and environmental > 50:
            comparison["recommendation"] = "Promising candidate - comparable to established quats. Consider experimental validation."
        elif efficacy > 60 and safety > 50:
            comparison["recommendation"] = "Moderate potential - may need optimization for safety or efficacy."
        elif efficacy > 50:
            comparison["recommendation"] = "Marginal activity predicted - structural modifications recommended."
        else:
            comparison["recommendation"] = "Low predicted activity - consider alternative scaffolds."

        # Add context about what scores mean
        comparison["score_interpretation"] = {
            "efficacy": self._interpret_efficacy(efficacy),
            "safety": self._interpret_safety(safety),
            "environmental": self._interpret_environmental(environmental)
        }

        return comparison

    def _interpret_efficacy(self, score: float) -> str:
        """Interpret efficacy score"""
        if score > 80:
            return "Excellent - comparable to best reference quats"
        elif score > 60:
            return "Good - typical of commercial disinfectants"
        elif score > 40:
            return "Moderate - may require higher concentrations"
        else:
            return "Low - limited antimicrobial potential"

    def _interpret_safety(self, score: float) -> str:
        """Interpret safety score"""
        if score > 80:
            return "Excellent safety profile expected"
        elif score > 60:
            return "Acceptable safety - typical for disinfectants"
        elif score > 40:
            return "Moderate concerns - careful application needed"
        else:
            return "Significant safety concerns predicted"

    def _interpret_environmental(self, score: float) -> str:
        """Interpret environmental score"""
        if score > 80:
            return "Excellent environmental profile"
        elif score > 60:
            return "Acceptable - moderate environmental impact"
        elif score > 40:
            return "Concerns about persistence or aquatic toxicity"
        else:
            return "Significant environmental concerns"

    def get_statistics(self) -> Dict:
        """Get summary statistics about reference compounds"""
        mic_s_aureus_values = [c.mic_s_aureus[0] for c in self.compounds.values() if c.mic_s_aureus]
        mic_e_coli_values = [c.mic_e_coli[0] for c in self.compounds.values() if c.mic_e_coli]
        ld50_values = [c.ld50_oral_rat for c in self.compounds.values() if c.ld50_oral_rat]

        return {
            "compound_count": len(self.compounds),
            "mic_s_aureus": {
                "min": min(mic_s_aureus_values) if mic_s_aureus_values else None,
                "max": max(c.mic_s_aureus[1] for c in self.compounds.values() if c.mic_s_aureus),
                "count": len(mic_s_aureus_values)
            },
            "mic_e_coli": {
                "min": min(mic_e_coli_values) if mic_e_coli_values else None,
                "max": max(c.mic_e_coli[1] for c in self.compounds.values() if c.mic_e_coli),
                "count": len(mic_e_coli_values)
            },
            "ld50_oral_rat": {
                "min": min(ld50_values) if ld50_values else None,
                "max": max(ld50_values) if ld50_values else None,
                "count": len(ld50_values)
            }
        }
