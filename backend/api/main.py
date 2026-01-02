"""
Quat Generator Pro - FastAPI Backend
Main application entry point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from api.routes import molecules, generator, export, status
from database.connection import init_db, close_db
from generator.engine import GeneratorEngine
from data import ChEMBLFetcher, ReferenceDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global generator engine instance
generator_engine: GeneratorEngine | None = None

# Global data fetchers
chembl_fetcher: ChEMBLFetcher | None = None
reference_db: ReferenceDatabase | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global generator_engine, chembl_fetcher, reference_db

    # Startup
    logger.info("Starting Quat Generator Pro backend...")

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Initialize reference database (curated compounds - fast, no network)
    reference_db = ReferenceDatabase()
    await reference_db.initialize()
    logger.info(f"Reference database initialized: {reference_db.compound_count} compounds")

    # Initialize ChEMBL fetcher (experimental data from ChEMBL)
    chembl_fetcher = ChEMBLFetcher(cache_dir="data/chembl_cache")
    await chembl_fetcher.initialize()
    logger.info(f"ChEMBL fetcher initialized: {chembl_fetcher.compound_count} compounds cached")

    # Initialize generator engine
    generator_engine = GeneratorEngine()
    await generator_engine.initialize()
    logger.info("Generator engine initialized")

    # Store in app state for access in routes
    app.state.generator = generator_engine
    app.state.chembl_fetcher = chembl_fetcher
    app.state.reference_db = reference_db

    yield

    # Shutdown
    logger.info("Shutting down...")
    if generator_engine:
        await generator_engine.shutdown()
    await close_db()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Quat Generator Pro",
    description="AI-powered quaternary ammonium compound designer",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS for Electron frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "app://.*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(molecules.router, prefix="/api/molecules", tags=["molecules"])
app.include_router(generator.router, prefix="/api/generator", tags=["generator"])
app.include_router(export.router, prefix="/api/export", tags=["export"])
app.include_router(status.router, prefix="/api/status", tags=["status"])


@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "name": "Quat Generator Pro",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "generator": "ready" if generator_engine and generator_engine.is_ready else "initializing",
        "chembl_fetcher": "ready" if chembl_fetcher and chembl_fetcher.is_ready else "initializing",
        "reference_db": "ready" if reference_db and reference_db.is_ready else "initializing"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
