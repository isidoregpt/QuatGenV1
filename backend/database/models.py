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
    """Generated molecule with scores and properties"""
    __tablename__ = "molecules"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    smiles = Column(String(500), nullable=False, unique=True, index=True)
    canonical_smiles = Column(String(500), nullable=True)
    inchi_key = Column(String(27), nullable=True, index=True)
    
    efficacy_score = Column(Float, nullable=False, default=0.0)
    safety_score = Column(Float, nullable=False, default=0.0)
    environmental_score = Column(Float, nullable=False, default=0.0)
    sa_score = Column(Float, nullable=False, default=0.0)
    combined_score = Column(Float, nullable=False, default=0.0)
    
    molecular_weight = Column(Float, nullable=True)
    logp = Column(Float, nullable=True)
    tpsa = Column(Float, nullable=True)
    hbd = Column(Integer, nullable=True)
    hba = Column(Integer, nullable=True)
    rotatable_bonds = Column(Integer, nullable=True)
    chain_length = Column(Integer, nullable=True)
    num_quat_n = Column(Integer, nullable=True)
    counterion = Column(String(10), nullable=True)
    
    is_valid_quat = Column(Boolean, default=True)
    is_valid_smiles = Column(Boolean, default=True)
    is_pareto = Column(Boolean, default=False, index=True)
    pareto_rank = Column(Integer, nullable=True)
    is_starred = Column(Boolean, default=False, index=True)
    notes = Column(Text, nullable=True)
    
    generation_run_id = Column(Integer, ForeignKey("generation_runs.id"), nullable=True)
    generation_step = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    properties = relationship("MoleculeProperty", back_populates="molecule", cascade="all, delete-orphan")
    generation_run = relationship("GenerationRun", back_populates="molecules")
    
    __table_args__ = (
        Index("ix_molecules_scores", "efficacy_score", "safety_score", "environmental_score"),
        Index("ix_molecules_pareto_starred", "is_pareto", "is_starred"),
        Index("ix_molecules_combined", "combined_score"),
    )
    
    def to_dict(self):
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
    """Extended properties for molecules"""
    __tablename__ = "molecule_properties"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    molecule_id = Column(Integer, ForeignKey("molecules.id", ondelete="CASCADE"), nullable=False)
    property_name = Column(String(100), nullable=False)
    property_category = Column(String(50), nullable=True)
    value_float = Column(Float, nullable=True)
    value_string = Column(String(500), nullable=True)
    value_int = Column(Integer, nullable=True)
    source = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    
    molecule = relationship("Molecule", back_populates="properties")
    
    __table_args__ = (
        Index("ix_molecule_properties_lookup", "molecule_id", "property_name"),
        UniqueConstraint("molecule_id", "property_name", name="uq_molecule_property"),
    )


class GenerationRun(Base):
    """Metadata for a generation run"""
    __tablename__ = "generation_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    target_molecules = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    weight_efficacy = Column(Float, default=0.4)
    weight_safety = Column(Float, default=0.3)
    weight_environmental = Column(Float, default=0.2)
    weight_sa = Column(Float, default=0.1)
    constraints_json = Column(Text, nullable=True)
    molecules_generated = Column(Integer, default=0)
    molecules_valid = Column(Integer, default=0)
    pareto_size = Column(Integer, default=0)
    best_efficacy = Column(Float, nullable=True)
    best_safety = Column(Float, nullable=True)
    best_environmental = Column(Float, nullable=True)
    best_combined = Column(Float, nullable=True)
    started_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)
    elapsed_seconds = Column(Float, nullable=True)
    status = Column(String(20), default="running")
    error_message = Column(Text, nullable=True)
    
    molecules = relationship("Molecule", back_populates="generation_run")


class ScoringCache(Base):
    """Cache for scoring predictions"""
    __tablename__ = "scoring_cache"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    smiles = Column(String(500), nullable=False, unique=True, index=True)
    efficacy_score = Column(Float, nullable=True)
    safety_score = Column(Float, nullable=True)
    environmental_score = Column(Float, nullable=True)
    sa_score = Column(Float, nullable=True)
    efficacy_details = Column(Text, nullable=True)
    safety_details = Column(Text, nullable=True)
    environmental_details = Column(Text, nullable=True)
    model_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    hit_count = Column(Integer, default=0)
