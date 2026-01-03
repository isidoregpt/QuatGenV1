# Quat Generator Pro - Configuration Guide

## Environment Variables

Create a `.env` file in the backend directory:

```env
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=sqlite+aiosqlite:///./quat_generator.db

# Models
PRETRAINED_MODEL=Franso/reinvent_171M_prior
ENCODER_MODEL=DeepChem/ChemBERTa-77M-MLM
DEVICE=cuda  # or cpu

# Generation
DEFAULT_BATCH_SIZE=64
MAX_MOLECULES_PER_RUN=1000
USE_PRETRAINED=true
USE_RL_FINETUNING=true

# Scoring
USE_ADMET_MODELS=true
ADMET_LAZY_LOAD=true
USE_MOLECULAR_ENCODER=true

# Caching
MODEL_CACHE_DIR=~/.cache/huggingface
CHEMBL_CACHE_DIR=data/chembl_cache
```

## Configuration Classes

### GenerationConfig

```python
@dataclass
class GenerationConfig:
    # Model settings
    use_pretrained: bool = True
    pretrained_model_name: str = "Franso/reinvent_171M_prior"
    device: str = "cuda"

    # Generation parameters
    batch_size: int = 64
    max_length: int = 128
    temperature: float = 1.0

    # RL settings
    use_rl_finetuning: bool = True
    rl_learning_rate: float = 1e-4
    rl_sigma: float = 60.0
```

### ScoringConfig

```python
@dataclass
class ScoringConfig:
    # Encoder
    use_molecular_encoder: bool = True
    encoder_model_name: str = "DeepChem/ChemBERTa-77M-MLM"
    encoder_pooling: str = "mean"

    # ADMET
    use_admet_models: bool = True
    admet_lazy_load: bool = True
    admet_models: List[str] = ["herg", "ames", "ld50", "dili"]

    # Weights
    efficacy_weight: float = 0.35
    safety_weight: float = 0.30
    environmental_weight: float = 0.20
    sa_weight: float = 0.15
```

### FilterConfig

```python
@dataclass
class FilterConfig:
    # Validity
    require_valid_smiles: bool = True
    require_quaternary_nitrogen: bool = True

    # Property ranges
    min_mw: float = 150.0
    max_mw: float = 800.0
    min_logp: float = -2.0
    max_logp: float = 10.0

    # Filters
    apply_pains_filter: bool = True
    apply_brenk_filter: bool = True
    diversity_threshold: float = 0.7
```

## Model Downloads

Models are downloaded automatically on first use. To pre-download:

```bash
python -c "
from huggingface_hub import snapshot_download

# Generator
snapshot_download('Franso/reinvent_171M_prior')

# Encoder
snapshot_download('DeepChem/ChemBERTa-77M-MLM')

# ADMET models
for model in ['herg', 'ames', 'ld50', 'dili', 'bbb', 'solubility']:
    snapshot_download(f'ChemFM/admet_{model}')
"
```

## GPU Configuration

### CUDA Setup

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Set specific GPU
export CUDA_VISIBLE_DEVICES=0
```

### Memory Management

For limited GPU memory:

```env
DEFAULT_BATCH_SIZE=32
ADMET_LAZY_LOAD=true
```

## Logging

Configure logging in `backend/logging_config.py`:

```python
LOGGING_CONFIG = {
    "version": 1,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "quat_generator.log",
            "level": "DEBUG"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}
```
