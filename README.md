# Quat Generator Pro

**AI-powered quaternary ammonium compound designer for next-generation disinfectants**

A desktop application for chemists to design novel quaternary ammonium compounds (quats) optimized for:
- **Efficacy**: Antimicrobial activity against bacteria, fungi, and viruses
- **Safety**: Low human toxicity (acute, dermal, ocular)
- **Environment**: Biodegradability, low aquatic toxicity, minimal bioaccumulation
- **Synthesizability**: Practical synthetic accessibility scores

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ELECTRON FRONTEND                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Generator  │  │   Results   │  │  Molecule   │  │   Export    │    │
│  │  Controls   │  │   Table     │  │   Detail    │  │   Modal     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │ HTTP/WebSocket
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      API Routes                                  │   │
│  │  /generate  /molecules  /scores  /export  /status               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │  Generator   │  │   Scoring    │  │   Database   │                 │
│  │   Engine     │  │   Pipeline   │  │   (SQLite)   │                 │
│  │  (RL+AIS)    │  │  (QSAR+ML)   │  │              │                 │
│  └──────────────┘  └──────────────┘  └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

### For Chemists
- **SMILES-centric interface**: Primary view is a sortable table of SMILES strings
- **Multi-objective optimization**: Balance efficacy, safety, and environmental impact
- **Pareto frontier visualization**: See trade-offs between objectives
- **Export formats**: CSV, SDF, PDF reports with 2D structures
- **Quat validation**: Automatic verification of quaternary nitrogen presence

### Technical
- **SMI+AIS tokenization**: Hybrid SMILES + Atom-In-SMILES encoding for better molecular generation
- **Reinforcement learning**: Policy gradient optimization toward multi-objective rewards
- **QSAR scoring**: EPA EPI Suite-equivalent predictions (BIOWIN, ECOSAR, BCF)
- **64-core parallelism**: Designed for high-end workstations
- **GPU acceleration**: PyTorch models on RTX 6000 (48GB VRAM)

## Requirements

### Hardware (Recommended)
- CPU: 32+ cores (designed for 64-core Threadripper)
- RAM: 64GB+ (designed for 128GB)
- GPU: NVIDIA RTX with 24GB+ VRAM
- Storage: 500GB+ SSD

### Software
- Windows 11 / Ubuntu 22.04+
- Python 3.11+
- Node.js 20+
- CUDA 12.0+ (for GPU acceleration)

## Installation

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download models (first run only)
python scripts/download_models.py
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Database Initialization

```bash
cd backend
python -m database.init_db
```

## Usage

### Start Backend Server

```bash
cd backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Start Frontend (Development)

```bash
cd frontend
npm run dev
```

### Build Desktop App

```bash
cd frontend
npm run build
npm run electron:build
```

## Project Structure

```
quat-generator/
├── frontend/                 # Electron + React + TypeScript
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── hooks/            # Custom hooks (useBackend, useGenerator)
│   │   ├── utils/            # Utilities (formatting, validation)
│   │   └── types/            # TypeScript interfaces
│   ├── public/               # Static assets
│   └── electron/             # Electron main process
│
├── backend/                  # Python FastAPI
│   ├── api/                  # REST endpoints
│   ├── generator/            # RL engine, policy network, tokenizer
│   ├── scoring/              # QSAR models (efficacy, safety, environmental)
│   ├── database/             # SQLite models and queries
│   ├── export/               # CSV, SDF, PDF generation
│   └── models/               # Trained model weights
│
├── data/                     # SQLite database, cached predictions
├── scripts/                  # Setup and utility scripts
├── docs/                     # Documentation
└── tests/                    # Test suites
```

## Scoring Models

### Efficacy Score (0-100%)
- Predicted MIC against gram-positive bacteria
- Predicted MIC against gram-negative bacteria  
- Predicted antifungal activity
- CMC (critical micelle concentration) estimation
- Membrane disruption potential

### Safety Score (0-100%)
- Acute oral toxicity (LD50 prediction)
- Skin irritation potential
- Eye irritation potential
- Respiratory sensitization risk

### Environmental Score (0-100%)
- Ready biodegradability (BIOWIN models 1-7)
- Aquatic toxicity (fish, daphnia, algae LC50/EC50)
- Bioconcentration factor (BCF)
- Persistence half-life

### Synthetic Accessibility (0-100%)
- RDKit SA score (inverted and normalized)
- Availability of starting materials
- Number of synthetic steps estimation

## Data Sources

### Training Data
- **ChEMBL**: Quaternary ammonium compounds with MIC data
- **PubChem10M**: General molecular vocabulary
- **EPA ADBAC Reports**: Toxicity data for benzalkonium chlorides

### QSAR Models
- **BIOWIN**: Biodegradability (7 models)
- **ECOSAR**: Aquatic ecotoxicity
- **Custom models**: Fine-tuned on quat-specific data

## API Reference

### Generate Molecules
```
POST /api/generate
{
  "num_molecules": 100,
  "constraints": {
    "min_mw": 200,
    "max_mw": 600,
    "min_chain_length": 8,
    "max_chain_length": 18,
    "require_quat": true
  },
  "weights": {
    "efficacy": 0.4,
    "safety": 0.3,
    "environmental": 0.2,
    "sa_score": 0.1
  }
}
```

### Get Molecules
```
GET /api/molecules?limit=100&offset=0&pareto_only=true&min_efficacy=50
```

### Export Results
```
POST /api/export
{
  "molecule_ids": [1, 2, 3],
  "format": "pdf",  // csv, sdf, pdf
  "include_structures": true
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest` (backend), `npm test` (frontend)
5. Submit a pull request

## License

MIT License - see LICENSE file

## Citation

If you use this software in your research, please cite:

```bibtex
@software{quat_generator_pro,
  title = {Quat Generator Pro: AI-Powered Quaternary Ammonium Compound Designer},
  year = {2024},
  url = {https://github.com/yourusername/quat-generator}
}
```

## Acknowledgments

- SMI+AIS tokenization based on [AIS-Drug-Opt](https://github.com/herim-han/AIS-Drug-Opt)
- QSAR methodology inspired by EPA EPI Suite
- ChEMBL database for training data
