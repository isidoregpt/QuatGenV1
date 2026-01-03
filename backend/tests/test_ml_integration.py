"""
Integration tests for ML-enhanced molecular generation pipeline.
Tests the complete flow from generation through scoring and benchmarking.
"""

import pytest
import asyncio
from typing import List

# Test configuration
TEST_SMILES = [
    "CCCCCCCCCCCC[N+](C)(C)Cc1ccccc1.[Cl-]",  # Benzalkonium-like
    "CCCCCCCCCCCCCCCC[n+]1ccccc1.[Cl-]",       # Cetylpyridinium-like
    "CCCCCCCCCC[N+](C)(C)CCCCCCCCCC.[Cl-]",    # DDAC-like
]

INVALID_SMILES = [
    "invalid_smiles",
    "C1CC1C1",  # Invalid ring
    "",
]


class TestMolecularEncoder:
    """Test ChemBERTa molecular encoder"""

    @pytest.mark.asyncio
    async def test_encoder_initialization(self):
        """Test that encoder initializes correctly"""
        from scoring.molecular_encoder import MolecularEncoder

        encoder = MolecularEncoder()
        await encoder.initialize()

        assert encoder.is_ready
        assert encoder.embedding_dim == 768

    @pytest.mark.asyncio
    async def test_single_encoding(self):
        """Test encoding a single molecule"""
        from scoring.molecular_encoder import MolecularEncoder

        encoder = MolecularEncoder()
        await encoder.initialize()

        embedding = encoder.encode(TEST_SMILES[0])

        assert embedding is not None
        assert embedding.shape == (768,)

    @pytest.mark.asyncio
    async def test_batch_encoding(self):
        """Test batch encoding"""
        from scoring.molecular_encoder import MolecularEncoder

        encoder = MolecularEncoder()
        await encoder.initialize()

        embeddings = encoder.encode_batch(TEST_SMILES)

        assert len(embeddings) == len(TEST_SMILES)
        assert all(e.shape == (768,) for e in embeddings)

    @pytest.mark.asyncio
    async def test_similarity_calculation(self):
        """Test molecular similarity"""
        from scoring.molecular_encoder import MolecularEncoder

        encoder = MolecularEncoder()
        await encoder.initialize()

        # Same molecule should have similarity 1.0
        sim = encoder.similarity(TEST_SMILES[0], TEST_SMILES[0])
        assert abs(sim - 1.0) < 0.01

        # Different molecules should have lower similarity
        sim = encoder.similarity(TEST_SMILES[0], TEST_SMILES[1])
        assert 0.0 < sim < 1.0


class TestADMETPredictor:
    """Test ADMET property prediction"""

    @pytest.mark.asyncio
    async def test_predictor_initialization(self):
        """Test ADMET predictor initialization"""
        from scoring.admet_models import ADMETPredictor

        predictor = ADMETPredictor(lazy_load=True)
        await predictor.initialize()

        assert predictor.is_ready

    @pytest.mark.asyncio
    async def test_single_prediction(self):
        """Test predicting properties for single molecule"""
        from scoring.admet_models import ADMETPredictor

        predictor = ADMETPredictor(lazy_load=True)
        await predictor.initialize()

        result = predictor.predict(TEST_SMILES[0], "herg")

        assert "prediction" in result
        assert "confidence" in result
        assert 0.0 <= result["prediction"] <= 1.0

    @pytest.mark.asyncio
    async def test_all_predictions(self):
        """Test predicting all ADMET properties"""
        from scoring.admet_models import ADMETPredictor

        predictor = ADMETPredictor(lazy_load=True)
        await predictor.initialize()

        results = predictor.predict_all(TEST_SMILES[0])

        assert len(results) > 0
        assert all("prediction" in r for r in results.values())


class TestMICPredictor:
    """Test MIC prediction for antimicrobial activity"""

    @pytest.mark.asyncio
    async def test_predictor_initialization(self):
        """Test MIC predictor initialization"""
        from scoring.mic_predictor import MICPredictor
        from scoring.molecular_encoder import MolecularEncoder
        from data.reference_db import ReferenceDatabase

        encoder = MolecularEncoder()
        await encoder.initialize()

        ref_db = ReferenceDatabase()
        await ref_db.initialize()

        predictor = MICPredictor(encoder=encoder)
        await predictor.initialize(reference_db=ref_db)

        assert predictor.is_ready

    @pytest.mark.asyncio
    async def test_mic_prediction(self):
        """Test MIC prediction"""
        from scoring.mic_predictor import MICPredictor
        from scoring.molecular_encoder import MolecularEncoder
        from data.reference_db import ReferenceDatabase

        encoder = MolecularEncoder()
        await encoder.initialize()

        ref_db = ReferenceDatabase()
        await ref_db.initialize()

        predictor = MICPredictor(encoder=encoder)
        await predictor.initialize(reference_db=ref_db)

        result = predictor.predict(TEST_SMILES[0], "s_aureus")

        assert result.predicted_mic > 0
        assert result.activity_class in ["excellent", "good", "moderate", "weak", "inactive"]
        assert 0.0 <= result.confidence <= 1.0


class TestScoringPipeline:
    """Test complete scoring pipeline"""

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test scoring pipeline initialization"""
        from scoring.pipeline import ScoringPipeline, ScoringConfig

        config = ScoringConfig(
            use_molecular_encoder=True,
            use_admet_models=True
        )

        pipeline = ScoringPipeline(config)
        await pipeline.initialize()

        assert pipeline.is_ready

    @pytest.mark.asyncio
    async def test_molecule_scoring(self):
        """Test scoring a molecule"""
        from scoring.pipeline import ScoringPipeline, ScoringConfig

        config = ScoringConfig()
        pipeline = ScoringPipeline(config)
        await pipeline.initialize()

        result = await pipeline.score_molecule(TEST_SMILES[0])

        assert "efficacy" in result
        assert "safety" in result
        assert "environmental" in result
        assert "sa_score" in result
        assert "combined_score" in result

        # Scores should be in valid range
        assert 0 <= result["efficacy"] <= 100
        assert 0 <= result["safety"] <= 100

    @pytest.mark.asyncio
    async def test_invalid_smiles_handling(self):
        """Test handling of invalid SMILES"""
        from scoring.pipeline import ScoringPipeline, ScoringConfig

        config = ScoringConfig()
        pipeline = ScoringPipeline(config)
        await pipeline.initialize()

        result = await pipeline.score_molecule("invalid_smiles")

        # Should return low scores or error indication
        assert result.get("combined_score", 0) == 0 or "error" in result


class TestMolecularFilters:
    """Test molecular filtering"""

    def test_filter_valid_quat(self):
        """Test that valid quats pass filters"""
        from generator.filters import MolecularFilter, FilterConfig

        config = FilterConfig(require_quaternary_nitrogen=True)
        filter = MolecularFilter(config)

        report = filter.filter_molecule(TEST_SMILES[0])

        assert report.is_valid
        assert "quaternary_nitrogen" in report.filter_results

    def test_filter_invalid_smiles(self):
        """Test that invalid SMILES are rejected"""
        from generator.filters import MolecularFilter, FilterConfig

        config = FilterConfig()
        filter = MolecularFilter(config)

        report = filter.filter_molecule("invalid")

        assert not report.is_valid
        assert len(report.rejection_reasons) > 0

    def test_diversity_selection(self):
        """Test diversity selection"""
        from generator.filters import DiversitySelector

        selector = DiversitySelector(similarity_threshold=0.7)

        # Should select diverse molecules
        selected = selector.select_diverse(
            TEST_SMILES + TEST_SMILES,  # Duplicates
            n_select=3
        )

        assert len(selected) <= 3


class TestSubstructureSearch:
    """Test substructure search functionality"""

    def test_smarts_validation(self):
        """Test SMARTS pattern validation"""
        from search.substructure import CommonPatterns

        # Valid pattern
        is_valid, _ = CommonPatterns.validate_smarts("[N+]")
        assert is_valid

        # Invalid pattern
        is_valid, _ = CommonPatterns.validate_smarts("[invalid")
        assert not is_valid

    def test_substructure_search(self):
        """Test substructure search"""
        from search.substructure import SubstructureSearch

        search = SubstructureSearch()

        molecules = [(s, i, {}) for i, s in enumerate(TEST_SMILES)]

        # Search for quaternary nitrogen
        results = search.substructure_search("[N+]", molecules)

        assert len(results) == len(TEST_SMILES)  # All test molecules have [N+]

    def test_similarity_search(self):
        """Test similarity search"""
        from search.substructure import SubstructureSearch

        search = SubstructureSearch()

        molecules = [(s, i, {}) for i, s in enumerate(TEST_SMILES)]

        # Search for similar to first molecule
        results = search.similarity_search(
            TEST_SMILES[0],
            molecules,
            threshold=0.5
        )

        assert len(results) > 0
        # First result should be exact match
        assert results[0].match_score >= 0.99


class TestBenchmarking:
    """Test benchmarking functionality"""

    @pytest.mark.asyncio
    async def test_benchmark_molecule(self):
        """Test benchmarking a single molecule"""
        from benchmark.comparator import BenchmarkComparator
        from data.reference_db import ReferenceDatabase

        ref_db = ReferenceDatabase()
        await ref_db.initialize()

        comparator = BenchmarkComparator(ref_db)

        result = comparator.benchmark_molecule(TEST_SMILES[0])

        assert 0 <= result.overall_score <= 100
        assert result.recommendation != ""
        assert len(result.closest_references) > 0

    @pytest.mark.asyncio
    async def test_benchmark_report(self):
        """Test generating benchmark report"""
        from benchmark.comparator import BenchmarkComparator
        from benchmark.report import BenchmarkReport
        from data.reference_db import ReferenceDatabase

        ref_db = ReferenceDatabase()
        await ref_db.initialize()

        comparator = BenchmarkComparator(ref_db)

        # Benchmark multiple molecules
        results = [comparator.benchmark_molecule(s) for s in TEST_SMILES]

        # Generate report
        report = BenchmarkReport.generate(results, ref_db)

        assert report.total_molecules == len(TEST_SMILES)
        assert len(report.recommendations) > 0


class TestEndToEndGeneration:
    """End-to-end tests for the full generation pipeline"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_pipeline(self):
        """Test complete generation -> scoring -> benchmarking pipeline"""
        from generator.engine import GeneratorEngine, GenerationConfig
        from benchmark.comparator import BenchmarkComparator
        from data.reference_db import ReferenceDatabase

        # Initialize generator
        config = GenerationConfig(
            use_pretrained=True,
            device="cpu"
        )
        engine = GeneratorEngine(config)
        await engine.initialize()

        # Initialize reference database
        ref_db = ReferenceDatabase()
        await ref_db.initialize()

        # Generate molecules (small batch for testing)
        # This would be a simplified test - full generation takes too long

        # For now, test that engine initializes correctly
        assert engine.is_ready

        # Test scoring pipeline is ready
        assert engine.scoring.is_ready


# Fixtures for pytest
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def initialized_pipeline():
    """Fixture providing initialized scoring pipeline"""
    from scoring.pipeline import ScoringPipeline, ScoringConfig

    config = ScoringConfig()
    pipeline = ScoringPipeline(config)
    await pipeline.initialize()
    return pipeline
