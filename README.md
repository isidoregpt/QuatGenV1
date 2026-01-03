# Quat Generator Pro

**AI-Powered Molecular Design Platform for Quaternary Ammonium Antimicrobial Compounds**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 18+](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org/)

Quat Generator Pro is a machine learning platform that designs novel quaternary ammonium compounds (quats) optimized for antimicrobial efficacy, human safety, and environmental sustainability. Built for medicinal chemists and antimicrobial researchers, it combines state-of-the-art generative AI with comprehensive scoring pipelines.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Manual Installation](#manual-installation)
- [First Run Notes](#first-run-notes)
- [GPU Acceleration](#gpu-acceleration)
- [Architecture](#architecture)
- [Models & Scoring](#models--scoring)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

### Core Capabilities

- **ML-Powered Generation**: REINVENT-based molecular generation with reinforcement learning fine-tuning, trained on 750 million molecules from the ZINC database
- **Multi-Objective Optimization**: Simultaneously balance efficacy, safety, environmental impact, and synthetic accessibility with customizable weights
- **ADMET Prediction**: Comprehensive toxicity and pharmacokinetic predictions using ChemFM and ChemBERTa models
- **MIC Prediction**: Estimate minimum inhibitory concentrations against key pathogens (*S. aureus*, *E. coli*, *P. aeruginosa*, *C. albicans*)

### Analysis Tools

- **Substructure Search**: SMARTS-based pattern matching to find molecules with specific structural motifs
- **Benchmark Comparison**: Compare generated molecules against known quaternary ammonium disinfectants (BAC, DDAC, CPC, Cetrimide, Octenidine)
- **Structure Visualization**: Interactive 2D molecular depictions with atom/bond highlighting
- **Property Calculations**: Molecular weight, LogP, TPSA, rotatable bonds, hydrogen bond donors/acceptors

### Data Management

- **Export Options**: CSV, SDF, and PDF report generation
- **Session Persistence**: SQLite database stores all generated molecules and scoring results
- **Batch Processing**: Generate and score hundreds of molecules in parallel

---

## Quick Start

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.11+ | 3.11 or 3.12 |
| **Node.js** | 18+ LTS | 20 LTS |
| **RAM** | 8 GB | 16 GB |
| **Disk Space** | 4 GB | 6 GB |
| **GPU** | None (CPU works) | NVIDIA CUDA or Apple Silicon |

**Download Links:**
- Python: https://www.python.org/downloads/
- Node.js: https://nodejs.org/

### Windows Installation

```batch
:: 1. Clone the repository
git clone https://github.com/isidoregpt/QuatGenV1.git
cd QuatGenV1

:: 2. Run setup (installs all dependencies, creates directories)
setup_windows.bat

:: 3. Start the application
start_windows.bat
```

### macOS / Linux Installation

```bash
# 1. Clone the repository
git clone https://github.com/isidoregpt/QuatGenV1.git
cd QuatGenV1

# 2. Make scripts executable
chmod +x setup_mac.sh start_mac.sh

# 3. Run setup (installs all dependencies, creates directories)
./setup_mac.sh

# 4. Start the application
./start_mac.sh
```

### Access the Application

Once both servers are running:

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend UI** | http://localhost:5173 | Main application interface |
| **Backend API** | http://localhost:8000 | REST API server |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger documentation |
| **ReDoc** | http://localhost:8000/redoc | Alternative API documentation |

---

## Manual Installation

If the automated scripts don't work for your environment, follow these steps:

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create required data directories
mkdir -p data/chembl_cache models

# Return to root
cd ..
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Return to root
cd ..
```

### Running Manually

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

---

## First Run Notes

### Model Downloads

The first startup takes **10-15 minutes** because the application downloads pre-trained ML models from Hugging Face:

| Model | Size | Purpose |
|-------|------|---------|
| **ChemBERTa-77M-MLM** | ~300 MB | Molecular embeddings and feature extraction |
| **REINVENT-171M** | ~700 MB | Generative molecular design |
| **ADMET Models** | ~500 MB | Toxicity and ADME property prediction |
| **Total** | **~1.5 GB** | |

Models are cached locally after first download. Subsequent starts are fast (<30 seconds).

### What Happens on First Run

1. **Model Detection**: Backend checks for cached models in `~/.cache/huggingface/`
2. **Automatic Download**: Missing models are downloaded from Hugging Face Hub
3. **Model Loading**: Models are loaded into memory (CPU or GPU)
4. **Database Initialization**: SQLite database created at `backend/data/molecules.db`
5. **ChEMBL Cache**: Reference antimicrobial data cached for MIC predictions

### Progress Indicators

The backend logs show download progress:
```
INFO: Downloading ChemBERTa-77M-MLM... 45%
INFO: Downloading REINVENT model... 78%
INFO: Loading ADMET predictors...
INFO: All models loaded successfully
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## GPU Acceleration

GPU acceleration significantly speeds up molecule generation and scoring.

### NVIDIA GPU (CUDA)

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support (example for CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU is detected:
```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Your GPU name
```

### Apple Silicon (M1/M2/M3/M4)

PyTorch automatically uses Metal Performance Shaders (MPS) acceleration on Apple Silicon. No additional setup required.

Verify MPS is available:
```python
import torch
print(torch.backends.mps.is_available())  # Should print True
```

### CPU-Only Mode

If you encounter GPU errors, force CPU mode by setting in your environment:
```bash
export CUDA_VISIBLE_DEVICES=""  # Linux/macOS
set CUDA_VISIBLE_DEVICES=       # Windows
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            REACT FRONTEND                                    │
│                         (TypeScript + Vite + Tailwind)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Generator  │  │   Search    │  │  Benchmark  │  │   Results Table     │ │
│  │  Controls   │  │   Panel     │  │   Panel     │  │   + Molecule Cards  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ HTTP REST API
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI BACKEND                                    │
│                            (Python 3.11+)                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         API Routes                                   │   │
│  │  /generator  /molecules  /search  /benchmark  /visualization /export│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Generator   │  │   Scoring    │  │  ML Models   │  │   Database   │    │
│  │   Engine     │  │   Pipeline   │  │ (HuggingFace)│  │   (SQLite)   │    │
│  │  (REINVENT)  │  │ (ADMET+MIC)  │  │              │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│         │                  │                  │                              │
│         ▼                  ▼                  ▼                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      RDKit Cheminformatics                           │   │
│  │    SMILES parsing, property calculation, substructure matching       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Request**: Frontend sends generation parameters to `/api/generator/start`
2. **Molecule Generation**: REINVENT model generates SMILES strings
3. **Validation**: RDKit validates chemical structures
4. **Scoring Pipeline**: Each molecule scored across 4 dimensions
5. **Database Storage**: Results saved to SQLite
6. **Response**: Scored molecules returned to frontend

---

## Models & Scoring

### Machine Learning Models

| Component | Model | Parameters | Training Data | Source |
|-----------|-------|------------|---------------|--------|
| **Generator** | REINVENT | 171M | 750M ZINC molecules | HuggingFace |
| **Encoder** | ChemBERTa-77M-MLM | 77M | PubChem compounds | DeepChem |
| **ADMET** | ChemFM suite | Various | ChEMBL bioactivity | HuggingFace |

### Scoring Dimensions

Each generated molecule receives four scores (0-100%):

#### Efficacy Score
Predicts antimicrobial effectiveness:
- MIC predictions against *S. aureus*, *E. coli*, *P. aeruginosa*, *C. albicans*
- Critical Micelle Concentration (CMC) estimation
- Membrane disruption potential
- Cationic charge density

#### Safety Score
Predicts human safety profile:
- **hERG inhibition**: Cardiac toxicity risk
- **AMES mutagenicity**: Genotoxicity potential
- **LD50 prediction**: Acute toxicity
- **DILI risk**: Drug-induced liver injury
- **Skin sensitization**: Dermal toxicity

#### Environmental Score
Predicts ecological impact:
- **Biodegradability**: Ready biodegradation (OECD 301)
- **Aquatic toxicity**: LC50 fish, EC50 daphnia
- **Bioconcentration Factor (BCF)**: Bioaccumulation potential
- **Persistence**: Environmental half-life

#### Synthetic Accessibility Score
Estimates ease of synthesis:
- **SA Score**: Ertl synthetic accessibility (1-10 → 0-100%)
- **Reaction step estimation**
- **Starting material availability**
- **Structural complexity**

### Combined Score

The combined score is a weighted average:
```
Combined = (Efficacy × w₁) + (Safety × w₂) + (Environmental × w₃) + (SA × w₄)
```

Default weights: Efficacy=0.35, Safety=0.30, Environmental=0.20, SA=0.15

---

## API Reference

### Generation Endpoints

#### Start Generation
```http
POST /api/generator/start
Content-Type: application/json

{
  "num_molecules": 100,
  "constraints": {
    "min_mw": 200,
    "max_mw": 600,
    "min_logp": -2,
    "max_logp": 8,
    "require_quaternary_nitrogen": true
  },
  "weights": {
    "efficacy": 0.35,
    "safety": 0.30,
    "environmental": 0.20,
    "synthesizability": 0.15
  }
}
```

#### Check Generation Status
```http
GET /api/generator/status

Response:
{
  "status": "running",
  "progress": 67,
  "molecules_generated": 67,
  "molecules_requested": 100,
  "elapsed_seconds": 45
}
```

#### Stop Generation
```http
POST /api/generator/stop
```

### Molecule Endpoints

#### List Molecules
```http
GET /api/molecules?limit=100&offset=0&sort_by=combined_score&order=desc

Response:
{
  "molecules": [...],
  "total": 523,
  "limit": 100,
  "offset": 0
}
```

#### Get Molecule Details
```http
GET /api/molecules/{molecule_id}

Response:
{
  "id": "mol_abc123",
  "smiles": "CCCCCCCCCCCC[N+](C)(C)Cc1ccccc1.[Cl-]",
  "scores": {
    "efficacy": 78.5,
    "safety": 82.1,
    "environmental": 65.3,
    "synthesizability": 71.2,
    "combined": 74.8
  },
  "properties": {
    "molecular_weight": 340.5,
    "logp": 4.2,
    "tpsa": 0.0,
    "hbd": 0,
    "hba": 0,
    "rotatable_bonds": 12
  },
  "predictions": {
    "mic_s_aureus": 2.5,
    "mic_e_coli": 8.0,
    "herg_inhibition": 0.15,
    "ames_positive": 0.08
  }
}
```

#### Render Structure
```http
POST /api/molecules/render
Content-Type: application/json

{
  "smiles": "CCCCCCCCCCCC[N+](C)(C)Cc1ccccc1.[Cl-]",
  "width": 400,
  "height": 300,
  "highlight_atoms": [12, 13, 14]
}

Response: PNG image (binary)
```

### Search Endpoints

#### Substructure Search
```http
POST /api/search/substructure
Content-Type: application/json

{
  "smarts": "[N+](C)(C)(C)",
  "max_results": 50
}

Response:
{
  "matches": [...],
  "count": 23,
  "query_smarts": "[N+](C)(C)(C)"
}
```

#### Similarity Search
```http
POST /api/search/similarity
Content-Type: application/json

{
  "smiles": "CCCCCCCCCCCC[N+](C)(C)Cc1ccccc1",
  "threshold": 0.7,
  "max_results": 20
}
```

### Benchmark Endpoints

#### Get Reference Compounds
```http
GET /api/benchmark/references

Response:
{
  "references": [
    {
      "name": "Benzalkonium Chloride (BAC)",
      "smiles": "CCCCCCCCCCCC[N+](C)(C)Cc1ccccc1.[Cl-]",
      "category": "First-generation quat",
      "typical_use": "Surface disinfectant, preservative"
    },
    ...
  ]
}
```

#### Benchmark Single Molecule
```http
POST /api/benchmark/molecule
Content-Type: application/json

{
  "smiles": "CCCCCCCCCCCC[N+](C)(C)Cc1ccccc1.[Cl-]"
}

Response:
{
  "molecule_scores": {...},
  "comparisons": [
    {
      "reference": "BAC",
      "similarity": 0.95,
      "efficacy_delta": +2.3,
      "safety_delta": -1.5
    }
  ]
}
```

### Export Endpoints

#### Export to CSV
```http
POST /api/export/csv
Content-Type: application/json

{
  "molecule_ids": ["mol_1", "mol_2", "mol_3"],
  "include_properties": true,
  "include_predictions": true
}

Response: CSV file (text/csv)
```

#### Export to SDF
```http
POST /api/export/sdf
Content-Type: application/json

{
  "molecule_ids": ["mol_1", "mol_2", "mol_3"]
}

Response: SDF file (chemical/x-mdl-sdfile)
```

#### Export PDF Report
```http
POST /api/export/pdf
Content-Type: application/json

{
  "molecule_ids": ["mol_1", "mol_2", "mol_3"],
  "include_structures": true,
  "include_radar_charts": true
}

Response: PDF file (application/pdf)
```

---

## Project Structure

```
QuatGenV1/
├── backend/                      # Python FastAPI backend
│   ├── api/                      # REST API layer
│   │   ├── main.py               # FastAPI application entry
│   │   └── routes/               # Route modules
│   │       ├── generator.py      # Generation endpoints
│   │       ├── molecules.py      # Molecule CRUD
│   │       ├── search.py         # Substructure/similarity search
│   │       ├── benchmark.py      # Reference comparisons
│   │       ├── visualization.py  # Structure rendering
│   │       └── export.py         # CSV/SDF/PDF export
│   ├── generator/                # Molecule generation engine
│   │   ├── reinvent.py           # REINVENT model wrapper
│   │   ├── constraints.py        # Chemical constraints
│   │   └── sampler.py            # Sampling strategies
│   ├── scoring/                  # Multi-objective scoring
│   │   ├── pipeline.py           # Scoring orchestration
│   │   ├── efficacy.py           # Efficacy scoring
│   │   ├── safety.py             # Safety scoring
│   │   ├── environmental.py      # Environmental scoring
│   │   ├── synthesizability.py   # SA scoring
│   │   ├── mic_predictor.py      # MIC prediction model
│   │   └── admet.py              # ADMET predictions
│   ├── benchmark/                # Reference compound comparison
│   │   ├── references.py         # Known quat database
│   │   └── comparator.py         # Comparison logic
│   ├── data/                     # Data files
│   │   ├── chembl_cache/         # Cached ChEMBL data
│   │   └── molecules.db          # SQLite database
│   ├── database/                 # Database layer
│   │   ├── connection.py         # SQLite connection
│   │   ├── models.py             # SQLAlchemy models
│   │   └── crud.py               # CRUD operations
│   ├── search/                   # Chemical search
│   │   ├── substructure.py       # SMARTS matching
│   │   └── similarity.py         # Fingerprint similarity
│   ├── visualization/            # Structure rendering
│   │   └── renderer.py           # RDKit 2D depiction
│   ├── models/                   # Downloaded ML models
│   └── requirements.txt          # Python dependencies
│
├── frontend/                     # React TypeScript frontend
│   ├── src/
│   │   ├── components/           # React components
│   │   │   ├── Generator/        # Generation controls
│   │   │   ├── MoleculeTable/    # Results table
│   │   │   ├── MoleculeCard/     # Detail cards
│   │   │   ├── SearchPanel/      # Search interface
│   │   │   ├── BenchmarkPanel/   # Benchmark view
│   │   │   └── Charts/           # Radar/bar charts
│   │   ├── hooks/                # Custom React hooks
│   │   ├── types/                # TypeScript types
│   │   ├── api/                  # API client
│   │   └── App.tsx               # Main application
│   ├── public/                   # Static assets
│   ├── package.json              # Node.js dependencies
│   ├── vite.config.ts            # Vite configuration
│   ├── tailwind.config.js        # Tailwind CSS config
│   └── tsconfig.json             # TypeScript config
│
├── docs/                         # Documentation
│   ├── CHEMIST_GUIDE.md          # Guide for chemists
│   ├── API_REFERENCE.md          # API documentation
│   ├── CONFIGURATION.md          # Configuration options
│   └── DEVELOPMENT.md            # Developer guide
│
├── setup_windows.bat             # Windows setup script
├── start_windows.bat             # Windows startup script
├── setup_mac.sh                  # macOS/Linux setup script
├── start_mac.sh                  # macOS/Linux startup script
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## Configuration

### Backend Configuration

Environment variables (set in `.env` or shell):

```bash
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Database
DATABASE_URL=sqlite:///./data/molecules.db

# Model Settings
MODEL_DEVICE=cuda  # or "cpu" or "mps"
MODEL_CACHE_DIR=~/.cache/huggingface

# Generation Defaults
DEFAULT_NUM_MOLECULES=100
MAX_MOLECULES_PER_REQUEST=1000

# Scoring Weights (default)
WEIGHT_EFFICACY=0.35
WEIGHT_SAFETY=0.30
WEIGHT_ENVIRONMENTAL=0.20
WEIGHT_SYNTHESIZABILITY=0.15
```

### Frontend Configuration

Edit `frontend/.env`:

```bash
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

---

## Troubleshooting

### Installation Issues

| Problem | Solution |
|---------|----------|
| **"Python not found"** | Add Python to PATH or use full path (`/usr/bin/python3`) |
| **"Node not found"** | Add Node.js to PATH or use full path |
| **"pip: command not found"** | Use `python -m pip` instead |
| **"Permission denied"** (macOS/Linux) | Run `chmod +x setup_mac.sh start_mac.sh` |

### Runtime Issues

| Problem | Solution |
|---------|----------|
| **"Port 8000 already in use"** | Kill process: `lsof -ti:8000 | xargs kill` or use `--port 8001` |
| **"Port 5173 already in use"** | Kill process or use `npm run dev -- --port 3000` |
| **"Module not found"** | Activate virtual environment before running backend |
| **CORS errors in browser** | Ensure backend is running at http://localhost:8000 |

### Model Issues

| Problem | Solution |
|---------|----------|
| **Models fail to download** | Check internet connection; try VPN if Hugging Face is blocked |
| **"CUDA out of memory"** | Reduce batch size or switch to CPU: `MODEL_DEVICE=cpu` |
| **"MPS not available"** | Update to macOS 12.3+ and PyTorch 1.12+ |
| **Slow generation on CPU** | Expected; GPU provides 10-50x speedup |

### Database Issues

| Problem | Solution |
|---------|----------|
| **"Database locked"** | Close other connections; restart backend |
| **Corrupt database** | Delete `backend/data/molecules.db` and restart |

### Getting Help

1. Check the [API documentation](http://localhost:8000/docs) for endpoint details
2. Review logs in the terminal running the backend
3. Open an issue on GitHub with error messages and steps to reproduce

---

## Documentation

| Document | Description |
|----------|-------------|
| [CHEMIST_GUIDE.md](docs/CHEMIST_GUIDE.md) | Comprehensive guide for non-technical chemists |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | Complete API documentation |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Configuration options and environment variables |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | Guide for developers and contributors |

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
cd backend
pip install -r requirements-dev.txt  # includes pytest, black, flake8

# Run tests
pytest

# Format code
black .
flake8
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

### Software & Libraries

- [RDKit](https://www.rdkit.org/) - Cheminformatics toolkit
- [Hugging Face](https://huggingface.co/) - Model hosting and transformers
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [React](https://reactjs.org/) - Frontend framework
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS

### Data Sources

- [ChEMBL](https://www.ebi.ac.uk/chembl/) - Antimicrobial activity data
- [ZINC](https://zinc.docking.org/) - Molecular database for training
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/) - Chemical structure data

### Models

- [REINVENT](https://github.com/MolecularAI/REINVENT) - Molecular generation architecture
- [ChemBERTa](https://github.com/deepchem/deepchem) - Molecular embeddings
- [ChemFM](https://huggingface.co/ChemFM) - ADMET prediction models

### Research

This tool implements concepts from:
- Olivecrona, M. et al. "Molecular de novo design through deep reinforcement learning" (2017)
- Chithrananda, S. et al. "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction" (2020)

---

## Value Proposition

For university quaternary ammonium researchers, Quat Generator Pro provides:

| Benefit | Estimated Value |
|---------|-----------------|
| **Time Savings** | 100-200 hours/year |
| **Cost Avoidance** | $8,000-70,000/year |
| **Scaffold Exploration** | 4-5x increase |
| **Total Annual Value** | $16,500-110,000 |

*Based on typical academic research lab workflows and compound synthesis costs.*

---

**Questions?** Open an issue or contact the maintainers.

**Happy molecule designing!** 🧪
