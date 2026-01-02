"""
ChEMBL data fetcher for quaternary ammonium compounds with antimicrobial activity

Fetches real experimental MIC data from ChEMBL database for quaternary ammonium
compounds tested against various microorganisms.
"""

import asyncio
import logging
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class QuatCompound:
    """Represents a quaternary ammonium compound with activity data from ChEMBL"""

    chembl_id: str
    smiles: str
    canonical_smiles: Optional[str] = None
    name: Optional[str] = None
    molecular_weight: Optional[float] = None

    # Activity data from ChEMBL
    activities: List[Dict] = field(default_factory=list)

    # Aggregated MIC values by organism
    mic_values: Dict[str, List[float]] = field(default_factory=dict)

    # Best (lowest) MIC per organism (lower = more potent)
    best_mic: Dict[str, float] = field(default_factory=dict)

    def add_activity(self, activity: Dict):
        """Add an activity record and update MIC aggregations"""
        self.activities.append(activity)

        # Extract organism and MIC if available
        organism = activity.get("target_organism", "Unknown")
        standard_type = activity.get("standard_type", "")
        standard_value = activity.get("standard_value")
        standard_units = activity.get("standard_units", "")

        # Only track MIC values in compatible units
        if standard_type in ["MIC", "MIC50", "MIC90"] and standard_value is not None:
            if standard_units in ["nM", "uM", "ug/mL", "ug.mL-1", "mg/L"]:
                try:
                    mic_value = float(standard_value)

                    if organism not in self.mic_values:
                        self.mic_values[organism] = []
                    self.mic_values[organism].append(mic_value)

                    # Update best MIC (lower is better)
                    if organism not in self.best_mic or mic_value < self.best_mic[organism]:
                        self.best_mic[organism] = mic_value
                except (ValueError, TypeError):
                    pass

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "chembl_id": self.chembl_id,
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "name": self.name,
            "molecular_weight": self.molecular_weight,
            "activities": self.activities,
            "mic_values": self.mic_values,
            "best_mic": self.best_mic
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "QuatCompound":
        """Create from dictionary"""
        return cls(
            chembl_id=data["chembl_id"],
            smiles=data["smiles"],
            canonical_smiles=data.get("canonical_smiles"),
            name=data.get("name"),
            molecular_weight=data.get("molecular_weight"),
            activities=data.get("activities", []),
            mic_values=data.get("mic_values", {}),
            best_mic=data.get("best_mic", {})
        )


class ChEMBLFetcher:
    """
    Fetches quaternary ammonium compound data from ChEMBL database.

    Provides access to real experimental antimicrobial activity data
    for training and validation of molecular generation models.
    """

    # SMARTS pattern for quaternary ammonium (positively charged nitrogen with 4 bonds)
    QUAT_SMARTS = "[N+;X4]"

    # Target organisms for antimicrobial activity
    TARGET_ORGANISMS = [
        "Staphylococcus aureus",
        "Escherichia coli",
        "Pseudomonas aeruginosa",
        "Candida albicans",
        "Bacillus subtilis",
        "Enterococcus faecalis",
        "Klebsiella pneumoniae",
        "Salmonella",
        "Listeria monocytogenes",
        "Streptococcus",
    ]

    # Known quaternary ammonium disinfectants to seed search
    KNOWN_QUATS = [
        "CHEMBL578",      # Benzalkonium chloride
        "CHEMBL1354",     # Cetylpyridinium chloride
        "CHEMBL1201135",  # Didecyldimethylammonium chloride
        "CHEMBL1236088",  # Benzethonium chloride
        "CHEMBL447964",   # Cetrimonium bromide
        "CHEMBL1201247",  # Domiphen bromide
        "CHEMBL1523",     # Dequalinium chloride
    ]

    def __init__(self, cache_dir: str = "data/chembl_cache"):
        """
        Initialize the ChEMBL fetcher.

        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        self.compounds: Dict[str, QuatCompound] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._client_initialized = False
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        """Check if fetcher is ready"""
        return self._is_ready

    @property
    def compound_count(self) -> int:
        """Total number of compounds loaded"""
        return len(self.compounds)

    @property
    def compounds_with_mic_count(self) -> int:
        """Number of compounds with MIC data"""
        return len([c for c in self.compounds.values() if c.mic_values])

    def _init_client(self):
        """Lazy initialization of ChEMBL client"""
        if self._client_initialized:
            return True

        try:
            from chembl_webresource_client.new_client import new_client
            self.molecule_client = new_client.molecule
            self.activity_client = new_client.activity
            self.target_client = new_client.target
            self.assay_client = new_client.assay
            self._client_initialized = True
            logger.info("ChEMBL client initialized successfully")
            return True
        except ImportError:
            logger.error("chembl_webresource_client not installed. Run: pip install chembl_webresource_client")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ChEMBL client: {e}")
            return False

    async def initialize(self) -> bool:
        """
        Initialize fetcher and load cache if available.

        Returns:
            True if initialization successful
        """
        try:
            os.makedirs(self.cache_dir, exist_ok=True)

            # Try to load from cache first
            cache_file = os.path.join(self.cache_dir, "quat_compounds.json")
            if os.path.exists(cache_file):
                try:
                    await self._load_cache(cache_file)
                    logger.info(f"Loaded {len(self.compounds)} compounds from cache")
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")

            self._is_ready = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ChEMBL fetcher: {e}")
            return False

    async def fetch_all(self, force_refresh: bool = False) -> Dict[str, QuatCompound]:
        """
        Fetch all quaternary ammonium compounds with antimicrobial activity.

        Args:
            force_refresh: If True, ignore cache and fetch fresh data

        Returns:
            Dictionary mapping ChEMBL IDs to QuatCompound objects
        """
        cache_file = os.path.join(self.cache_dir, "quat_compounds.json")

        # Check cache validity
        if not force_refresh and os.path.exists(cache_file):
            cache_age = datetime.now().timestamp() - os.path.getmtime(cache_file)
            if cache_age < 86400 * 7:  # Cache valid for 7 days
                logger.info("Using cached data (less than 7 days old)")
                if not self.compounds:
                    await self._load_cache(cache_file)
                return self.compounds

        # Initialize ChEMBL client
        if not self._init_client():
            logger.warning("ChEMBL client not available, using cached data only")
            return self.compounds

        # Fetch from known quats first
        logger.info("Fetching known quaternary ammonium compounds...")
        await self._fetch_known_quats()

        # Search for compounds with antimicrobial activity containing quat nitrogen
        logger.info("Searching for quat compounds with antimicrobial activity...")
        await self._fetch_antimicrobial_quats()

        # Save to cache
        await self._save_cache(cache_file)
        logger.info(f"Fetched {len(self.compounds)} total compounds")

        return self.compounds

    async def _fetch_known_quats(self):
        """Fetch data for known quaternary ammonium disinfectants"""
        loop = asyncio.get_event_loop()

        for chembl_id in self.KNOWN_QUATS:
            try:
                # Fetch molecule info
                mol_data = await loop.run_in_executor(
                    self._executor,
                    lambda cid=chembl_id: list(self.molecule_client.filter(chembl_id=cid))
                )

                if not mol_data:
                    logger.debug(f"No data found for {chembl_id}")
                    continue

                mol = mol_data[0]
                structures = mol.get("molecule_structures") or {}
                smiles = structures.get("canonical_smiles")

                if not smiles:
                    logger.debug(f"No SMILES for {chembl_id}")
                    continue

                properties = mol.get("molecule_properties") or {}
                compound = QuatCompound(
                    chembl_id=chembl_id,
                    smiles=smiles,
                    canonical_smiles=smiles,
                    name=mol.get("pref_name"),
                    molecular_weight=properties.get("full_mwt")
                )

                # Fetch activity data
                activities = await loop.run_in_executor(
                    self._executor,
                    lambda cid=chembl_id: list(self.activity_client.filter(
                        molecule_chembl_id=cid,
                        standard_type__in=["MIC", "MIC50", "MIC90", "IC50", "EC50"]
                    ).only([
                        "molecule_chembl_id", "target_organism", "standard_type",
                        "standard_value", "standard_units", "assay_description"
                    ]))
                )

                for activity in activities:
                    compound.add_activity(activity)

                self.compounds[chembl_id] = compound
                logger.info(f"Fetched {chembl_id}: {compound.name or 'Unknown'}, {len(activities)} activities")

            except Exception as e:
                logger.error(f"Error fetching {chembl_id}: {e}")

    async def _fetch_antimicrobial_quats(self):
        """Search for quaternary ammonium compounds with antimicrobial assays"""
        loop = asyncio.get_event_loop()

        # Limit to first few organisms to avoid overloading
        for organism in self.TARGET_ORGANISMS[:5]:
            try:
                logger.info(f"Searching activities against {organism}...")

                # Search for MIC activities against this organism
                activities = await loop.run_in_executor(
                    self._executor,
                    lambda org=organism: list(self.activity_client.filter(
                        target_organism__icontains=org,
                        standard_type__in=["MIC", "MIC50", "MIC90"]
                    ).only([
                        "molecule_chembl_id", "target_organism", "standard_type",
                        "standard_value", "standard_units", "canonical_smiles"
                    ])[:500])  # Limit results per organism
                )

                logger.info(f"Found {len(activities)} activities against {organism}")

                # Filter for quaternary ammonium compounds
                for activity in activities:
                    smiles = activity.get("canonical_smiles", "")
                    chembl_id = activity.get("molecule_chembl_id")

                    if not smiles or not chembl_id:
                        continue

                    # Check if it's a quat (contains [N+] or [n+] for pyridinium)
                    if "[N+]" not in smiles and "[n+]" not in smiles:
                        continue

                    # Add or update compound
                    if chembl_id not in self.compounds:
                        self.compounds[chembl_id] = QuatCompound(
                            chembl_id=chembl_id,
                            smiles=smiles,
                            canonical_smiles=smiles
                        )

                    self.compounds[chembl_id].add_activity(activity)

            except Exception as e:
                logger.error(f"Error searching {organism}: {e}")

    async def _save_cache(self, cache_file: str):
        """Save compounds to cache file"""
        cache_data = {
            "fetched_at": datetime.now().isoformat(),
            "version": "1.0",
            "compound_count": len(self.compounds),
            "compounds": {
                cid: c.to_dict()
                for cid, c in self.compounds.items()
            }
        }

        try:
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Saved {len(self.compounds)} compounds to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    async def _load_cache(self, cache_file: str):
        """Load compounds from cache file"""
        with open(cache_file, "r") as f:
            cache_data = json.load(f)

        for cid, data in cache_data.get("compounds", {}).items():
            self.compounds[cid] = QuatCompound.from_dict(data)

    def get_compounds_with_mic(self, organism: Optional[str] = None) -> List[QuatCompound]:
        """
        Get compounds that have MIC data, optionally filtered by organism.

        Args:
            organism: Filter by organism name (case-insensitive partial match)

        Returns:
            List of compounds with MIC data
        """
        results = []
        for compound in self.compounds.values():
            if not compound.mic_values:
                continue

            if organism:
                # Check if any organism matches
                if any(organism.lower() in org.lower() for org in compound.mic_values.keys()):
                    results.append(compound)
            else:
                results.append(compound)

        return results

    def get_training_data(self) -> List[Dict]:
        """
        Get data formatted for model training: SMILES + MIC values.

        Returns:
            List of dictionaries with smiles, organism, mic_value, chembl_id
        """
        training_data = []

        for compound in self.compounds.values():
            for organism, mic_list in compound.mic_values.items():
                for mic in mic_list:
                    training_data.append({
                        "smiles": compound.smiles,
                        "organism": organism,
                        "mic_value": mic,
                        "chembl_id": compound.chembl_id,
                        "name": compound.name
                    })

        return training_data

    def get_organisms(self) -> List[str]:
        """Get list of all organisms with MIC data"""
        organisms = set()
        for compound in self.compounds.values():
            organisms.update(compound.mic_values.keys())
        return sorted(list(organisms))

    def get_smiles_list(self) -> List[str]:
        """Get list of all SMILES strings"""
        return [c.smiles for c in self.compounds.values() if c.smiles]

    def get_compound(self, chembl_id: str) -> Optional[QuatCompound]:
        """Get a specific compound by ChEMBL ID"""
        return self.compounds.get(chembl_id)

    def search_by_smiles(self, smiles: str) -> Optional[QuatCompound]:
        """Search for a compound by SMILES"""
        for compound in self.compounds.values():
            if compound.smiles == smiles or compound.canonical_smiles == smiles:
                return compound
        return None
