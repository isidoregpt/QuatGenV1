# Architecture

## Overview
Quat 2.0 AI Lab is split into:
- **Compute plane (local)**: `agent/` runs the 24/7 virtual design loop (generation + scoring + RL).
- **Control plane (cloud)**: `vercel/` (UI/API) + `firebase/` (storage/rules/functions).

## Representation Layer (SMILES-first + optional SMI+AIS)
Quat 2.0 AI Lab is SMILES-native. We optionally enable **SMI+AIS(N)**, a hybrid representation that replaces a subset of atom tokens with AIS tokens encoding local chemical environment (central atom; ring; neighbors).

Vocabulary construction:
- Convert corpus to AIS tokens
- Count frequency
- Keep top-N AIS tokens, represent all others as standard SMILES

Default configuration: **SMI+AIS(100)**.

## Safety note
This repository contains *virtual design and scoring* scaffolding only and intentionally excludes any real-world synthesis guidance.
