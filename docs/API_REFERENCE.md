# Quat Generator Pro - API Reference

## Overview

The Quat Generator Pro API provides endpoints for generating, scoring, searching, and benchmarking quaternary ammonium compounds.

Base URL: `http://localhost:8000/api`

## Authentication

Currently, no authentication is required. Future versions may implement API keys.

---

## Endpoints

### Generator Endpoints

#### POST /generator/start
Start a molecule generation run.

**Request Body:**
```json
{
  "num_molecules": 100,
  "batch_size": 64,
  "weights": {
    "efficacy": 0.4,
    "safety": 0.3,
    "environmental": 0.15,
    "synthesis": 0.15
  },
  "constraints": {
    "min_mw": 200,
    "max_mw": 600,
    "require_quat": true
  }
}
```

**Response:**
```json
{
  "status": "started",
  "job_id": "gen_12345"
}
```

#### GET /generator/status
Get current generation status.

**Response:**
```json
{
  "is_running": true,
  "molecules_generated": 50,
  "current_batch": 2,
  "total_batches": 4,
  "best_scores": {
    "efficacy": 85.2,
    "safety": 72.1,
    "combined": 78.5
  }
}
```

#### GET /generator/rl/status
Get REINVENT RL training status.

**Response:**
```json
{
  "available": true,
  "training_active": false,
  "current_step": 150,
  "best_score": 0.82,
  "replay_buffer_size": 500
}
```

---

### Molecule Endpoints

#### GET /molecules
List all generated molecules.

**Query Parameters:**
- `limit` (int): Maximum results (default: 100)
- `offset` (int): Pagination offset
- `sort_by` (string): Sort field (combined_score, efficacy_score, etc.)
- `order` (string): asc or desc

#### GET /molecules/{id}
Get single molecule details.

**Response:**
```json
{
  "id": 1,
  "smiles": "CCCCCCCCCCCC[N+](C)(C)Cc1ccccc1.[Cl-]",
  "efficacy_score": 85.2,
  "safety_score": 72.1,
  "environmental_score": 65.0,
  "sa_score": 78.5,
  "combined_score": 75.2,
  "molecular_weight": 340.5,
  "logp": 4.2
}
```

#### GET /molecules/{id}/image
Get 2D structure image.

**Query Parameters:**
- `width` (int): Image width (default: 400)
- `height` (int): Image height (default: 300)
- `format` (string): "png" or "svg"
- `highlight_quat` (bool): Highlight quaternary nitrogen

**Response:** Binary image data

#### POST /molecules/render
Render any SMILES to image.

**Query Parameters:**
- `smiles` (string): SMILES to render
- `width`, `height`, `format`, `highlight_quat`: Same as above

**Response:**
```json
{
  "smiles": "...",
  "image_data_uri": "data:image/png;base64,...",
  "molecule_info": {
    "num_atoms": 25,
    "formula": "C20H36ClN",
    "has_quat_nitrogen": true
  }
}
```

#### GET /molecules/{id}/mic
Get MIC predictions for a molecule.

**Response:**
```json
{
  "molecule_id": 1,
  "predictions": {
    "s_aureus": {
      "mic": 4.2,
      "confidence": 0.75,
      "activity_class": "good"
    },
    "e_coli": {
      "mic": 16.5,
      "confidence": 0.72,
      "activity_class": "moderate"
    }
  }
}
```

#### GET /molecules/{id}/admet
Get ADMET property predictions.

**Response:**
```json
{
  "molecule_id": 1,
  "predictions": {
    "herg": {"prediction": 0.15, "risk": "low"},
    "ames": {"prediction": 0.08, "risk": "low"},
    "ld50": {"prediction": 2.8, "interpretation": "slightly toxic"}
  }
}
```

---

### Search Endpoints

#### GET /search/patterns
Get available SMARTS pattern templates.

**Response:**
```json
{
  "categories": ["basic", "scaffold", "functional"],
  "patterns": {
    "any_quat": {
      "smarts": "[N+,n+]",
      "description": "Any quaternary nitrogen",
      "category": "basic"
    }
  }
}
```

#### POST /search/substructure
Search molecules by substructure.

**Request Body:**
```json
{
  "smarts": "[N+]Cc1ccccc1",
  "max_results": 50,
  "include_image": true
}
```

**Response:**
```json
{
  "query": "[N+]Cc1ccccc1",
  "search_type": "substructure",
  "matches_found": 15,
  "results": [
    {
      "smiles": "...",
      "molecule_id": 5,
      "matched_atoms": [0, 1, 2],
      "properties": {"efficacy": 82}
    }
  ]
}
```

#### POST /search/similarity
Search molecules by similarity.

**Request Body:**
```json
{
  "smiles": "CCCCCCCCCCCC[N+](C)(C)Cc1ccccc1",
  "threshold": 0.7,
  "max_results": 20
}
```

---

### Benchmark Endpoints

#### POST /benchmark/molecule
Benchmark a molecule against references.

**Request Body:**
```json
{
  "smiles": "...",
  "predicted_scores": {"efficacy_score": 75}
}
```

**Response:**
```json
{
  "smiles": "...",
  "overall_score": 72.5,
  "recommendation": "Good candidate - comparable to established quats",
  "closest_references": [...],
  "property_comparisons": [...],
  "advantages": ["Improved safety profile"],
  "disadvantages": []
}
```

#### POST /benchmark/report
Generate full benchmark report.

**Response:**
```json
{
  "generated_at": "2024-01-15T10:30:00Z",
  "summary": {
    "total_molecules": 100,
    "avg_overall_score": 68.5,
    "top_candidates_count": 12
  },
  "scaffold_distribution": {...},
  "recommendations": [...]
}
```

---

## Error Responses

All endpoints return errors in this format:
```json
{
  "detail": "Error message description"
}
```

HTTP Status Codes:
- 400: Bad Request (invalid input)
- 404: Not Found
- 500: Internal Server Error
- 503: Service Unavailable (model not loaded)
