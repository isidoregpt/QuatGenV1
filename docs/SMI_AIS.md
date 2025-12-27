# SMI+AIS Representation

## What it is
SMI+AIS(N) is a hybrid string representation that preserves SMILES grammar while replacing some atom tokens with Atom-in-SMILES (AIS) tokens.

AIS token includes:
- central atom
- ring membership (R or !R)
- neighbor atoms (symbols)

## How we build the vocabulary
1. Convert a training SMILES corpus to AIS tokens.
2. Count token frequency.
3. Choose top-N frequent AIS tokens as the AIS vocabulary.
4. Tokenization replaces an atom token with its AIS token only if it is in the top-N set; otherwise it keeps standard SMILES.

## Recommended N
- Default: **N=100**
- Consider: N=150
- Avoid early: N=200 (can introduce sparsity)
