"""
Database models for Quat Generator Pro
SQLAlchemy ORM models
"""

from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, Text, ForeignKey,
    Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.connection import Base


class Molecule(Base):
    """
    Generated molecule with scores and properties
    
    This is the main table storing all generated quaternary ammonium compounds.
    """
    __tablename__ = "molecules"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Core identifiers
    smiles = Column(String(500), nullable=False, unique=True, index=True)
    canonical_smiles = Column(String(500), nullable=True)
    inchi_key = Column(String(27), nullable=True, index=True)
    
    # Multi-objective scores (0-100)
    efficacy_score = Column(Float, nullable=False, default=0.0)
    safety_score = Column(Float, nullable=False, default=0.0)
    environmental_score = Column(Float, nullable=False, default=0.0)
    sa_score = Column(Float, nullable=False, default=0.0)
    
    # Combined score (weighted sum)
    combined_score = Column(Float, nullable=False, default=0.0)
    
    # Basic molecular properties
    molecular_weight = Column(Float, nullable=True)
    logp = Column(Float, nullable=True)
    tpsa = Column(Float, nullable=True)  # Topological polar surface area
    hbd = Column(Integer, nullable=True)  # H-bond donors
    hba = Column(Integer, nullable=True)  # H-bond acceptors
    rotatable_bonds = Column(Integer, nullable=True)
    
    # Quat-specific properties
    chain_length = Column(Integer, nullable=True)  # Longest alkyl chain
    num_quat_n = Column(Integer, nullable=True)    # Number of quaternary nitrogens
    counterion = Column(String(10), nullable=True)  # Cl, Br, I, etc.
    
    # Validation flags
    is_valid_quat = Column(Boolean, default=True)
    is_valid_smiles = Column(Boolean, default=True)
    
    # Optimization flags
    is_pareto = Column(Boolean, default=False, index=True)
    pareto_rank = Column(Integer, nullable=True)
    
    # User flags
    is_starred = Column(Boolean, default=False, index=True)
    notes = Column(Text, nullable=True)
    
    # Generation metadata
    generation_run_id = Column(Integer, ForeignKey("generation_runs.id"), nullable=True)
    generation_step = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    properties = relationship("MoleculeProperty", back_populates="molecule", cascade="all, delete-orphan")
    generation_run = relationship("GenerationRun", back_populates="molecules")
    
    # Indexes for common queries
    __table_args__ = (
        Index("ix_molecules_scores", "efficacy_score", "safety_score", "environmental_score"),
        Index("ix_molecules_pareto_starred", "is_pareto", "is_starred"),
        Index("ix_molecules_combined", "combined_score"),
    )
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "smiles": self.smiles,
            "efficacy_score": self.efficacy_score,
            "safety_score": self.safety_score,
            "environmental_score": self.environmental_score,
            "sa_score": self.sa_score,
            "combined_score": self.combined_score,
            "molecular_weight": self.molecular_weight,
            "logp": self.logp,
            "chain_length": self.chain_length,
            "is_valid_quat": self.is_valid_quat,
            "is_pareto": self.is_pareto,
            "is_starred": self.is_starred,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class MoleculeProperty(Base):
    """
    Extended properties for molecules
    
    Stores additional calculated properties that don't fit in the main table.
    Uses key-value structure for flexibility.
    """
    __tablename__ = "molecule_properties"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    molecule_id = Column(Integer, ForeignKey("molecules.id", ondelete="CASCADE"), nullable=False)
    
    # Property identification
    property_name = Column(String(100), nullable=False)
    property_category = Column(String(50), nullable=True)  # efficacy, safety, environmental, etc.
    
    # Property value (store as string, parse based on type)
    value_float = Column(Float, nullable=True)
    value_string = Column(String(500), nullable=True)
    value_int = Column(Integer, nullable=True)
    
    # Metadata
    source = Column(String(100), nullable=True)  # Model or calculation source
    confidence = Column(Float, nullable=True)     # Prediction confidence if applicable
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    molecule = relationship("Molecule", back_populates="properties")
    
    __table_args__ = (
        Index("ix_molecule_properties_lookup", "molecule_id", "property_name"),
        UniqueConstraint("molecule_id", "property_name", name="uq_molecule_property"),
    )


class GenerationRun(Base):
    """
    Metadata for a generation run
    
    Tracks parameters and results for each generation session.
    """
    __tablename__ = "generation_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Run parameters
    target_molecules = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    
    # Objective weights
    weight_efficacy = Column(Float, default=0.4)
    weight_safety = Column(Float, default=0.3)
    weight_environmental = Column(Float, default=0.2)
    weight_sa = Column(Float, default=0.1)
    
    # Constraints (stored as JSON string)
    constraints_json = Column(Text, nullable=True)
    
    # Results
    molecules_generated = Column(Integer, default=0)
    molecules_valid = Column(Integer, default=0)
    pareto_size = Column(Integer, default=0)
    
    # Best scores achieved
    best_efficacy = Column(Float, nullable=True)
    best_safety = Column(Float, nullable=True)
    best_environmental = Column(Float, nullable=True)
    best_combined = Column(Float, nullable=True)
    
    # Timing
    started_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)
    elapsed_seconds = Column(Float, nullable=True)
    
    # Status
    status = Column(String(20), default="running")  # running, completed, stopped, error
    error_message = Column(Text, nullable=True)
    
    # Relationships
    molecules = relationship("Molecule", back_populates="generation_run")


class ScoringCache(Base):
    """
    Cache for scoring predictions
    
    Avoids redundant model inference for previously scored SMILES.
    """
    __tablename__ = "scoring_cache"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    smiles = Column(String(500), nullable=False, unique=True, index=True)
    
    # Cached scores
    efficacy_score = Column(Float, nullable=True)
    safety_score = Column(Float, nullable=True)
    environmental_score = Column(Float, nullable=True)
    sa_score = Column(Float, nullable=True)
    
    # Detailed predictions (JSON)
    efficacy_details = Column(Text, nullable=True)
    safety_details = Column(Text, nullable=True)
    environmental_details = Column(Text, nullable=True)
    
    # Cache metadata
    model_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    hit_count = Column(Integer, default=0)
