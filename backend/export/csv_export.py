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
    """
    Export molecules to CSV format
    
    Args:
        molecules: List of Molecule objects
        delimiter: CSV delimiter
        include_header: Include column headers
        columns: Specific columns to include (None = all)
        include_properties: Include molecular properties
        include_scores: Include score columns
    
    Returns:
        CSV string
    """
    output = io.StringIO()
    
    # Define columns
    all_columns = ["id", "smiles"]
    
    if include_scores:
        all_columns.extend([
            "efficacy_score",
            "safety_score", 
            "environmental_score",
            "sa_score",
            "combined_score"
        ])
    
    if include_properties:
        all_columns.extend([
            "molecular_weight",
            "logp",
            "chain_length",
            "is_valid_quat",
            "is_pareto",
            "is_starred"
        ])
    
    all_columns.append("created_at")
    
    # Filter columns if specified
    if columns:
        all_columns = [c for c in all_columns if c in columns]
    
    writer = csv.DictWriter(
        output,
        fieldnames=all_columns,
        delimiter=delimiter,
        extrasaction="ignore"
    )
    
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
