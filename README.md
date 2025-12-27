# Quat Generator Pro

**AI-powered quaternary ammonium compound designer for next-generation disinfectants**

A desktop application for chemists to design novel quaternary ammonium compounds (quats) optimized for:
- **Efficacy**: Antimicrobial activity against bacteria, fungi, and viruses
- **Safety**: Low human toxicity (acute, dermal, ocular)
- **Environment**: Biodegradability, low aquatic toxicity, minimal bioaccumulation
- **Synthesizability**: Practical synthetic accessibility scores

## Quick Start

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

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
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │  Generator   │  │   Scoring    │  │   Database   │                 │
│  │   Engine     │  │   Pipeline   │  │   (SQLite)   │                 │
│  │  (RL+AIS)    │  │  (QSAR+ML)   │  │              │                 │
│  └──────────────┘  └──────────────┘  └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## Scoring Models

- **Efficacy (0-100%)**: MIC predictions, CMC, membrane disruption
- **Safety (0-100%)**: Oral toxicity, skin/eye irritation, respiratory
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

### Get Molecules
```
GET /api/molecules?limit=100&pareto_only=true
```

### Export Results
```
POST /api/export/csv
POST /api/export/sdf
POST /api/export/pdf
```

## License

MIT License
