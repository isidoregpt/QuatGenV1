"""
Database connection management
SQLite with async support via aiosqlite
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from contextlib import asynccontextmanager
import os

# Database path
DB_PATH = os.environ.get("QUAT_DB_PATH", "data/quats.db")
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True
)

# Session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


class Base(DeclarativeBase):
    """Base class for all models"""
    pass


async def init_db():
    """Initialize database - create all tables"""
    from database.models import Molecule, MoleculeProperty, GenerationRun
    
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections"""
    await engine.dispose()


async def get_db() -> AsyncSession:
    """Dependency for getting database session"""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context():
    """Context manager for database session"""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db_stats() -> dict:
    """Get database statistics"""
    from sqlalchemy import text
    
    async with async_session() as session:
        result = await session.execute(
            text("SELECT COUNT(*) FROM molecules")
        )
        molecule_count = result.scalar() or 0
        
        db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
        
        return {
            "path": DB_PATH,
            "size_mb": round(db_size / (1024 * 1024), 2),
            "molecule_count": molecule_count,
            "connected": True
        }
