# Quat Generator Pro - Development Guide

## Project Structure

```
quat-generator-pro/
├── backend/
│   ├── api/
│   │   ├── main.py          # FastAPI application
│   │   └── routes/          # API endpoints
│   ├── generator/
│   │   ├── engine.py        # Generation engine
│   │   ├── pretrained_model.py
│   │   ├── reinvent.py      # RL training
│   │   └── filters.py       # Molecular filters
│   ├── scoring/
│   │   ├── pipeline.py      # Scoring orchestration
│   │   ├── efficacy.py      # MIC-based scoring
│   │   ├── safety.py        # ADMET-based scoring
│   │   ├── environmental.py
│   │   ├── sa_score.py      # Synthetic accessibility
│   │   ├── molecular_encoder.py
│   │   ├── admet_models.py
│   │   └── mic_predictor.py
│   ├── search/
│   │   └── substructure.py
│   ├── benchmark/
│   │   ├── comparator.py
│   │   └── report.py
│   ├── data/
│   │   ├── chembl_fetcher.py
│   │   └── reference_db.py
│   ├── visualization/
│   │   └── renderer.py
│   ├── database/
│   │   ├── models.py
│   │   ├── session.py
│   │   └── queries.py
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── molecules/
│   │   │   ├── search/
│   │   │   └── benchmark/
│   │   ├── services/
│   │   │   └── api.ts
│   │   └── styles/
│   └── package.json
└── docs/
```

## Development Setup

### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run with auto-reload
uvicorn api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Run tests
npm test

# Build for production
npm run build
```

## Adding New Features

### Adding a New Scorer

1. Create scorer in `scoring/`:

```python
# scoring/new_scorer.py
class NewScorer:
    async def initialize(self):
        ...

    async def score(self, smiles: str) -> Dict:
        ...
```

2. Register in pipeline:

```python
# scoring/pipeline.py
from scoring.new_scorer import NewScorer

class ScoringPipeline:
    async def initialize(self):
        self.new_scorer = NewScorer()
        await self.new_scorer.initialize()
```

3. Add API endpoint:

```python
# api/routes/molecules.py
@router.get("/{id}/new-score")
async def get_new_score(id: int):
    ...
```

### Adding a New API Endpoint

1. Add route in appropriate file:

```python
# api/routes/feature.py
from fastapi import APIRouter

router = APIRouter(prefix="/feature", tags=["feature"])

@router.get("/")
async def get_feature():
    return {"status": "ok"}
```

2. Register router in main.py:

```python
from api.routes import feature
app.include_router(feature.router, prefix="/api")
```

### Adding Frontend Components

1. Create component:

```tsx
// components/feature/FeatureComponent.tsx
export const FeatureComponent: React.FC<Props> = ({ ... }) => {
    return <div>...</div>;
};
```

2. Add API methods:

```typescript
// services/api.ts
export const featureApi = {
    getData: async () => {
        const response = await fetch(`${API_BASE}/feature`);
        return response.json();
    }
};
```

## Testing

### Backend Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ml_integration.py

# Run with coverage
pytest --cov=. --cov-report=html

# Skip slow tests
pytest -m "not slow"
```

### Frontend Tests

```bash
# Run tests
npm test

# Run with coverage
npm test -- --coverage
```

## Code Style

### Python

- Use Black for formatting
- Use isort for imports
- Use type hints
- Follow PEP 8

```bash
black backend/
isort backend/
mypy backend/
```

### TypeScript

- Use ESLint + Prettier
- Use strict TypeScript

```bash
npm run lint
npm run format
```

## Deployment

### Docker

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY backend/ .
RUN pip install -r requirements.txt

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0"]
```

### Production Settings

```env
DEBUG=false
LOG_LEVEL=WARNING
WORKERS=4
```
