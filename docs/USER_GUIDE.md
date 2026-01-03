# Quat Generator Pro - User Guide

## Introduction

Quat Generator Pro is an AI-powered tool for designing quaternary ammonium antimicrobial compounds. It uses state-of-the-art machine learning models to generate, score, and optimize molecular structures.

## Getting Started

### System Requirements

- Python 3.10+
- Node.js 18+
- 8GB RAM minimum (16GB recommended)
- GPU optional but recommended for faster generation

### Quick Start

1. Start the backend:
```bash
cd backend
pip install -r requirements.txt
uvicorn api.main:app --reload
```

2. Start the frontend:
```bash
cd frontend
npm install
npm run dev
```

3. Open http://localhost:5173 in your browser

---

## Features

### Molecule Generation

The generator creates novel quaternary ammonium compounds optimized for your specified objectives.

#### Setting Objectives

Adjust the weight sliders to prioritize different properties:

- **Efficacy** (0-100%): Prioritize antimicrobial activity
- **Safety** (0-100%): Prioritize low toxicity
- **Environmental** (0-100%): Prioritize biodegradability
- **Synthesis** (0-100%): Prioritize ease of synthesis

#### Generation Parameters

- **Number of Molecules**: How many to generate (10-1000)
- **Batch Size**: Processing batch size (affects memory usage)

#### Starting Generation

1. Set your objective weights
2. Configure constraints (optional)
3. Click "Start Generation"
4. Monitor progress in real-time

### Viewing Results

Generated molecules appear in a grid with:
- 2D structure image
- Scores for each objective
- Combined optimization score

Click any molecule to view details:
- Full-size structure
- ADMET predictions
- MIC predictions
- Benchmark comparison

### Searching Molecules

#### Substructure Search

Find molecules containing specific structural features:

1. Switch to Search tab
2. Select "Substructure Search"
3. Enter a SMARTS pattern or select from templates
4. Click Search

Common patterns:
- `[N+]` - Any quaternary nitrogen
- `[N+]Cc1ccccc1` - Benzylammonium
- `[n+]1ccccc1` - Pyridinium

#### Similarity Search

Find molecules similar to a reference:

1. Select "Similarity Search"
2. Enter reference SMILES
3. Adjust similarity threshold (50-100%)
4. Click Search

### Benchmarking

Compare your molecules to known antimicrobial quats.

#### Single Molecule Benchmark

1. Go to Benchmark tab
2. Enter SMILES or select molecule
3. View comparison to closest references
4. See predicted advantages/disadvantages

#### Batch Benchmark

1. Select "Batch Analysis"
2. Set minimum score threshold
3. Click "Run Batch Benchmark"
4. Review ranked results

#### Full Report

Generate comprehensive analysis:
- Scaffold distribution
- Comparison statistics
- Recommendations

---

## Interpreting Results

### Score Interpretation

| Score | Interpretation |
|-------|---------------|
| 80-100 | Excellent - prioritize for synthesis |
| 65-79 | Good - comparable to known quats |
| 50-64 | Moderate - may need optimization |
| 35-49 | Below average - significant changes needed |
| 0-34 | Poor - consider different scaffold |

### MIC Predictions

| MIC (µg/mL) | Activity Class |
|-------------|----------------|
| ≤2 | Excellent |
| 2-8 | Good |
| 8-32 | Moderate |
| 32-128 | Weak |
| >128 | Inactive |

### Safety Indicators

- **hERG**: Cardiac toxicity risk (lower is safer)
- **AMES**: Mutagenicity (lower is safer)
- **DILI**: Liver toxicity (lower is safer)
- **LD50**: Acute toxicity (higher is safer)

---

## Best Practices

1. **Start broad, then focus**: Begin with equal weights, then adjust based on results

2. **Use diversity filters**: Enable diversity to explore more chemical space

3. **Benchmark early**: Compare to references before synthesizing

4. **Check ADMET**: Review safety predictions for top candidates

5. **Validate structures**: Verify generated structures make chemical sense

---

## Troubleshooting

### Models not loading

First run downloads models (~2GB). Ensure:
- Internet connection available
- Sufficient disk space
- HuggingFace accessible

### Slow generation

- Reduce batch size if memory limited
- Enable GPU if available
- Reduce number of molecules

### Invalid structures generated

- Enable stricter filters
- Increase diversity threshold
- Check constraint settings
