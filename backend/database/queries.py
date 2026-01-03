"""
Database query functions
Reusable queries for molecule operations
"""

from sqlalchemy import select, update, delete, func, and_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from database.models import Molecule, MoleculeProperty, GenerationRun, ScoringCache


async def get_molecules(
    db: AsyncSession,
    limit: int = 100,
    offset: int = 0,
    filters: dict = None,
    sort_by: str = "created_at",
    sort_desc: bool = True
) -> tuple[list[Molecule], int]:
    """Get molecules with filtering and pagination"""
    query = select(Molecule)
    count_query = select(func.count(Molecule.id))
    
    if filters:
        conditions = []
        if filters.get("pareto_only"):
            conditions.append(Molecule.is_pareto == True)
        if filters.get("starred_only"):
            conditions.append(Molecule.is_starred == True)
        if filters.get("min_efficacy") is not None:
            conditions.append(Molecule.efficacy_score >= filters["min_efficacy"])
        if filters.get("min_safety") is not None:
            conditions.append(Molecule.safety_score >= filters["min_safety"])
        if filters.get("min_environmental") is not None:
            conditions.append(Molecule.environmental_score >= filters["min_environmental"])
        if filters.get("min_sa") is not None:
            conditions.append(Molecule.sa_score >= filters["min_sa"])
        if conditions:
            query = query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))
    
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    sort_column = getattr(Molecule, sort_by, Molecule.created_at)
    query = query.order_by(desc(sort_column) if sort_desc else asc(sort_column))
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    molecules = result.scalars().all()
    
    return molecules, total


async def get_molecule_by_id(db: AsyncSession, molecule_id: int) -> Optional[Molecule]:
    result = await db.execute(select(Molecule).where(Molecule.id == molecule_id))
    return result.scalar_one_or_none()


async def get_molecule_by_smiles(db: AsyncSession, smiles: str) -> Optional[Molecule]:
    result = await db.execute(select(Molecule).where(Molecule.smiles == smiles))
    return result.scalar_one_or_none()


async def get_molecules_by_ids(db: AsyncSession, ids: list[int]) -> list[Molecule]:
    result = await db.execute(select(Molecule).where(Molecule.id.in_(ids)))
    return result.scalars().all()


async def create_molecule(db: AsyncSession, molecule_data: dict) -> Molecule:
    molecule = Molecule(**molecule_data)
    db.add(molecule)
    await db.flush()
    return molecule


async def update_molecule_starred(db: AsyncSession, molecule_id: int, starred: bool) -> Optional[Molecule]:
    await db.execute(update(Molecule).where(Molecule.id == molecule_id).values(is_starred=starred))
    return await get_molecule_by_id(db, molecule_id)


async def delete_molecule(db: AsyncSession, molecule_id: int) -> bool:
    result = await db.execute(delete(Molecule).where(Molecule.id == molecule_id))
    return result.rowcount > 0


async def bulk_update_starred(db: AsyncSession, molecule_ids: list[int], starred: bool) -> int:
    result = await db.execute(update(Molecule).where(Molecule.id.in_(molecule_ids)).values(is_starred=starred))
    return result.rowcount


async def bulk_delete_molecules(db: AsyncSession, molecule_ids: list[int]) -> int:
    result = await db.execute(delete(Molecule).where(Molecule.id.in_(molecule_ids)))
    return result.rowcount


async def get_molecule_properties(db: AsyncSession, molecule_id: int) -> dict:
    result = await db.execute(select(MoleculeProperty).where(MoleculeProperty.molecule_id == molecule_id))
    properties = result.scalars().all()
    props_dict = {}
    for prop in properties:
        if prop.value_float is not None:
            props_dict[prop.property_name] = prop.value_float
        elif prop.value_int is not None:
            props_dict[prop.property_name] = prop.value_int
        else:
            props_dict[prop.property_name] = prop.value_string
    return props_dict


async def get_generation_info(db: AsyncSession, molecule_id: int) -> dict:
    molecule = await get_molecule_by_id(db, molecule_id)
    if not molecule or not molecule.generation_run_id:
        return {}
    result = await db.execute(select(GenerationRun).where(GenerationRun.id == molecule.generation_run_id))
    run = result.scalar_one_or_none()
    if not run:
        return {}
    return {
        "run_id": run.id,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "generation_step": molecule.generation_step,
        "weights": {"efficacy": run.weight_efficacy, "safety": run.weight_safety, "environmental": run.weight_environmental, "sa": run.weight_sa}
    }


async def get_similar_molecules(db: AsyncSession, molecule_id: int, limit: int = 10) -> list[Molecule]:
    molecule = await get_molecule_by_id(db, molecule_id)
    if not molecule:
        return []
    result = await db.execute(
        select(Molecule).where(
            and_(
                Molecule.id != molecule_id,
                Molecule.efficacy_score.between(molecule.efficacy_score - 10, molecule.efficacy_score + 10),
                Molecule.safety_score.between(molecule.safety_score - 10, molecule.safety_score + 10)
            )
        ).limit(limit)
    )
    return result.scalars().all()


async def get_molecule_stats(db: AsyncSession) -> dict:
    total = await db.execute(select(func.count(Molecule.id)))
    pareto = await db.execute(select(func.count(Molecule.id)).where(Molecule.is_pareto == True))
    starred = await db.execute(select(func.count(Molecule.id)).where(Molecule.is_starred == True))
    avg_scores = await db.execute(select(
        func.avg(Molecule.efficacy_score), func.avg(Molecule.safety_score),
        func.avg(Molecule.environmental_score), func.avg(Molecule.sa_score)
    ))
    avgs = avg_scores.one()
    max_scores = await db.execute(select(
        func.max(Molecule.efficacy_score), func.max(Molecule.safety_score),
        func.max(Molecule.environmental_score), func.max(Molecule.sa_score)
    ))
    maxs = max_scores.one()
    return {
        "total_molecules": total.scalar() or 0,
        "pareto_molecules": pareto.scalar() or 0,
        "starred_molecules": starred.scalar() or 0,
        "average_scores": {"efficacy": round(avgs[0] or 0, 2), "safety": round(avgs[1] or 0, 2), "environmental": round(avgs[2] or 0, 2), "sa": round(avgs[3] or 0, 2)},
        "max_scores": {"efficacy": round(maxs[0] or 0, 2), "safety": round(maxs[1] or 0, 2), "environmental": round(maxs[2] or 0, 2), "sa": round(maxs[3] or 0, 2)}
    }


async def update_pareto_frontier(db: AsyncSession) -> int:
    await db.execute(update(Molecule).values(is_pareto=False))
    result = await db.execute(select(Molecule))
    molecules = result.scalars().all()
    pareto_ids = []
    for mol in molecules:
        is_dominated = False
        for other in molecules:
            if other.id == mol.id:
                continue
            if (other.efficacy_score >= mol.efficacy_score and other.safety_score >= mol.safety_score and
                other.environmental_score >= mol.environmental_score and other.sa_score >= mol.sa_score and
                (other.efficacy_score > mol.efficacy_score or other.safety_score > mol.safety_score or
                 other.environmental_score > mol.environmental_score or other.sa_score > mol.sa_score)):
                is_dominated = True
                break
        if not is_dominated:
            pareto_ids.append(mol.id)
    if pareto_ids:
        await db.execute(update(Molecule).where(Molecule.id.in_(pareto_ids)).values(is_pareto=True))
    return len(pareto_ids)


async def get_all_molecules(db: AsyncSession, limit: int = 10000) -> list[Molecule]:
    """
    Get all molecules from database for search operations.

    Args:
        db: Database session
        limit: Maximum number of molecules to return

    Returns:
        List of Molecule objects
    """
    result = await db.execute(select(Molecule).limit(limit))
    return result.scalars().all()
