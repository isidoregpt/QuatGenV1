"""SDF Export for molecules"""

from typing import List


async def export_to_sdf(
    molecules: List,
    include_properties: bool = True,
    include_scores: bool = True
) -> str:
    """
    Export molecules to SDF (Structure-Data File) format
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return _export_sdf_simple(molecules, include_properties, include_scores)
    
    sdf_blocks = []
    
    for mol_obj in molecules:
        try:
            mol = Chem.MolFromSmiles(mol_obj.smiles)
            if mol is None:
                continue
            
            AllChem.Compute2DCoords(mol)
            mol.SetProp("_Name", f"QUAT_{mol_obj.id}")
            mol_block = Chem.MolToMolBlock(mol)
            
            data_fields = []
            if include_scores:
                data_fields.extend([
                    f">  <EFFICACY_SCORE>\n{mol_obj.efficacy_score:.1f}\n",
                    f">  <SAFETY_SCORE>\n{mol_obj.safety_score:.1f}\n",
                    f">  <ENVIRONMENTAL_SCORE>\n{mol_obj.environmental_score:.1f}\n",
                    f">  <SA_SCORE>\n{mol_obj.sa_score:.1f}\n",
                ])
            
            if include_properties:
                if mol_obj.molecular_weight:
                    data_fields.append(f">  <MOLECULAR_WEIGHT>\n{mol_obj.molecular_weight:.2f}\n")
                if mol_obj.logp:
                    data_fields.append(f">  <LOGP>\n{mol_obj.logp:.2f}\n")
            
            data_fields.append(f">  <SMILES>\n{mol_obj.smiles}\n")
            
            sdf_block = mol_block + "\n".join(data_fields) + "\n$$$$\n"
            sdf_blocks.append(sdf_block)
            
        except Exception:
            continue
    
    return "".join(sdf_blocks)


def _export_sdf_simple(molecules, include_properties, include_scores) -> str:
    """Fallback SDF export without RDKit"""
    blocks = []
    for mol_obj in molecules:
        block = f"QUAT_{mol_obj.id}\n\n\n  0  0  0  0  0  0  0  0  0  0  0 V2000\nM  END\n"
        block += f">  <SMILES>\n{mol_obj.smiles}\n\n$$$$\n"
        blocks.append(block)
    return "".join(blocks)
