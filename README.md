# Quat Generator Pro

AI-powered molecular design tool for quaternary ammonium antimicrobial compounds.

## Features

- **ML-Powered Generation**: Uses pretrained molecular transformers (REINVENT)
- **Multi-Objective Optimization**: Balance efficacy, safety, environment, synthesis
- **ADMET Prediction**: ChemFM models for toxicity and ADME properties
- **MIC Prediction**: Estimate antimicrobial activity against pathogens
- **Substructure Search**: SMARTS-based molecular search
- **Benchmarking**: Compare to known quaternary ammonium disinfectants
- **Structure Visualization**: 2D depictions with highlighting

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/quat-generator-pro.git
cd quat-generator-pro

# Start backend
cd backend
pip install -r requirements.txt
uvicorn api.main:app --reload

# Start frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Documentation

- [User Guide](docs/USER_GUIDE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Configuration](docs/CONFIGURATION.md)
- [Development](docs/DEVELOPMENT.md)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         REACT FRONTEND                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Generator  │  │   Search    │  │  Benchmark  │  │   Results   │    │
│  │  Controls   │  │   Panel     │  │   Panel     │  │   Table     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                   │ HTTP/WebSocket
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      API Routes                                  │   │
│  │  /generate  /molecules  /search  /benchmark  /visualization     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │  Generator   │  │   Scoring    │  │  ML Models   │                 │
│  │   Engine     │  │   Pipeline   │  │  (HuggingFace)│                 │
│  │  (REINVENT)  │  │  (ADMET+MIC) │  │              │                 │
│  └──────────────┘  └──────────────┘  └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## Models Used

| Component | Model | Source |
|-----------|-------|--------|
| Generator | REINVENT 171M | HuggingFace |
| Encoder | ChemBERTa-77M | DeepChem |
| ADMET | ChemFM suite | HuggingFace |

## Scoring Models

- **Efficacy (0-100%)**: MIC predictions, CMC, membrane disruption
- **Safety (0-100%)**: hERG, AMES, LD50, DILI predictions
- **Environmental (0-100%)**: Biodegradability, aquatic toxicity, BCF
- **SA Score (0-100%)**: Synthetic accessibility

## API Reference

### Generate Molecules
```
POST /api/generator/start
{
  "num_molecules": 100,
  "constraints": { "min_mw": 200, "max_mw": 600, ... },
  "weights": { "efficacy": 0.4, "safety": 0.3, ... }
}
```

### Search Molecules
```
POST /api/search/substructure
{
  "smarts": "[N+]Cc1ccccc1",
  "max_results": 50
}
```

### Benchmark Molecules
```
POST /api/benchmark/molecule
{
  "smiles": "CCCCCCCCCCCC[N+](C)(C)Cc1ccccc1.[Cl-]"
}
```

### Get Molecules
```
GET /api/molecules?limit=100&sort_by=combined_score
```

### Export Results
```
POST /api/export/csv
POST /api/export/sdf
POST /api/export/pdf
```

## Requirements

- Python 3.10+
- Node.js 18+
- 8GB RAM (16GB recommended)
- GPU optional

## License

MIT License - See LICENSE file
