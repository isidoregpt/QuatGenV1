from __future__ import annotations
import argparse
from pathlib import Path

from agent.smiles.vocab import AISVocab


def iter_smiles_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles-file", required=True, help="Text file with one SMILES per line")
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--out", required=True, help="Output JSON vocab path")
    args = ap.parse_args()

    vocab = AISVocab.build(iter_smiles_file(Path(args.smiles_file)), top_n=args.top_n)
    vocab.save(args.out)
    print(f"Saved AIS vocab: top_n={args.top_n} tokens={len(vocab.ais_tokens)} -> {args.out}")


if __name__ == "__main__":
    main()
