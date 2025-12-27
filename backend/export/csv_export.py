"""CSV Export for molecules"""

import csv
import io
from typing import List, Optional


async def export_to_csv(
    molecules: List,
    delimiter: str = ",",
    include_header: bool = True,
    columns: Optional[List[str]] = None,
    include_properties: bool = True,
    include_scores: bool = True
) -> str:
    output = io.StringIO()
    all_columns = ["id", "smiles"]
    if include_scores:
        all_columns.extend(["efficacy_score", "safety_score", "environmental_score", "sa_score", "combined_score"])
    if include_properties:
        all_columns.extend(["molecular_weight", "logp", "chain_length", "is_valid_quat", "is_pareto", "is_starred"])
    all_columns.append("created_at")
    if columns:
        all_columns = [c for c in all_columns if c in columns]
    writer = csv.DictWriter(output, fieldnames=all_columns, delimiter=delimiter, extrasaction="ignore")
    if include_header:
        writer.writeheader()
    for mol in molecules:
        row = {}
        for col in all_columns:
            value = getattr(mol, col, None)
            if col == "created_at" and value:
                value = value.isoformat()
            row[col] = value
        writer.writerow(row)
    return output.getvalue()
