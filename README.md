# Quat 2.0 AI Lab (SMILES-first, optional SMI+AIS)

This repository is a **full-stack skeleton** for a SMILES-native virtual design + scoring system.

## Architecture (high level)
- **agent/**: Local core (GPU) running inside WSL2 Ubuntu (Windows 11 host). Runs generation + scoring + RL.
- **vercel/**: Next.js control plane (UI + lightweight APIs). No GPU, no long-running training.
- **firebase/**: Firestore rules/indexes + optional Cloud Functions (aggregation/validation).
- **shared/**: Cross-cutting schemas/constants.
- **docs/**: Design documents and rationale.

## Safety Note
This is a **virtual design and scoring platform**. It **does not** provide synthesis procedures, recipes, or laboratory instructions.
