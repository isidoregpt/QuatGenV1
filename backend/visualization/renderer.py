"""
Molecular Structure Renderer

Generates 2D depictions of molecules using RDKit with customizable styling.
"""

import logging
import io
import base64
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available - structure rendering disabled")


class ImageFormat(Enum):
    PNG = "png"
    SVG = "svg"


@dataclass
class RenderConfig:
    """Configuration for molecule rendering"""
    # Image dimensions
    width: int = 400
    height: int = 300

    # Format
    format: ImageFormat = ImageFormat.PNG

    # Styling
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # RGBA white
    bond_line_width: float = 2.0
    atom_label_font_size: int = 16

    # Highlighting
    highlight_quat_nitrogen: bool = True
    quat_highlight_color: Tuple[float, float, float, float] = (0.0, 0.6, 0.9, 0.5)  # Blue
    highlight_atoms: List[int] = field(default_factory=list)
    highlight_bonds: List[int] = field(default_factory=list)
    highlight_color: Tuple[float, float, float, float] = (1.0, 0.8, 0.0, 0.5)  # Yellow

    # Atom coloring
    use_atom_colors: bool = True

    # 2D coordinates
    compute_coords: bool = True
    coord_gen_method: str = "rdkit"  # "rdkit" or "coordgen"

    # Annotations
    show_atom_indices: bool = False
    show_stereochemistry: bool = True

    # Grid options (for batch rendering)
    mols_per_row: int = 4
    legend_font_size: int = 12


class MoleculeRenderer:
    """
    Renders 2D molecular structures with customizable styling.
    """

    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self._is_ready = RDKIT_AVAILABLE

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def render_molecule(self,
                        smiles: str,
                        config: Optional[RenderConfig] = None,
                        highlight_smarts: Optional[str] = None) -> Optional[bytes]:
        """
        Render a single molecule to image bytes.

        Args:
            smiles: SMILES string
            config: Optional config override
            highlight_smarts: Optional SMARTS pattern to highlight

        Returns:
            Image bytes (PNG or SVG) or None if rendering fails
        """
        if not self._is_ready:
            return None

        cfg = config or self.config

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return None

            # Compute 2D coordinates
            if cfg.compute_coords:
                self._compute_2d_coords(mol, cfg.coord_gen_method)

            # Find atoms to highlight
            highlight_atoms = list(cfg.highlight_atoms)
            highlight_atom_colors = {}

            # Highlight quaternary nitrogens
            if cfg.highlight_quat_nitrogen:
                quat_atoms = self._find_quat_nitrogens(mol)
                for atom_idx in quat_atoms:
                    highlight_atoms.append(atom_idx)
                    highlight_atom_colors[atom_idx] = cfg.quat_highlight_color[:3]

            # Highlight custom SMARTS pattern
            if highlight_smarts:
                pattern = Chem.MolFromSmarts(highlight_smarts)
                if pattern:
                    matches = mol.GetSubstructMatches(pattern)
                    for match in matches:
                        for atom_idx in match:
                            if atom_idx not in highlight_atoms:
                                highlight_atoms.append(atom_idx)
                                highlight_atom_colors[atom_idx] = cfg.highlight_color[:3]

            # Create drawer
            if cfg.format == ImageFormat.SVG:
                drawer = rdMolDraw2D.MolDraw2DSVG(cfg.width, cfg.height)
            else:
                drawer = rdMolDraw2D.MolDraw2DCairo(cfg.width, cfg.height)

            # Configure drawing options
            opts = drawer.drawOptions()
            opts.bondLineWidth = cfg.bond_line_width
            opts.setBackgroundColour(cfg.background_color)
            opts.addAtomIndices = cfg.show_atom_indices

            # Draw molecule
            if highlight_atoms:
                drawer.DrawMolecule(
                    mol,
                    highlightAtoms=highlight_atoms,
                    highlightAtomColors=highlight_atom_colors if highlight_atom_colors else None,
                    highlightBonds=cfg.highlight_bonds if cfg.highlight_bonds else None
                )
            else:
                drawer.DrawMolecule(mol)

            drawer.FinishDrawing()

            # Get image data
            if cfg.format == ImageFormat.SVG:
                return drawer.GetDrawingText().encode('utf-8')
            else:
                return drawer.GetDrawingText()

        except Exception as e:
            logger.error(f"Rendering error for {smiles}: {e}")
            return None

    def render_molecule_base64(self,
                               smiles: str,
                               config: Optional[RenderConfig] = None,
                               highlight_smarts: Optional[str] = None) -> Optional[str]:
        """
        Render molecule and return as base64 encoded string.

        Returns:
            Base64 encoded image string or None
        """
        img_bytes = self.render_molecule(smiles, config, highlight_smarts)
        if img_bytes is None:
            return None

        return base64.b64encode(img_bytes).decode('utf-8')

    def render_molecule_data_uri(self,
                                 smiles: str,
                                 config: Optional[RenderConfig] = None,
                                 highlight_smarts: Optional[str] = None) -> Optional[str]:
        """
        Render molecule and return as data URI for embedding in HTML.

        Returns:
            Data URI string or None
        """
        cfg = config or self.config
        base64_img = self.render_molecule_base64(smiles, config, highlight_smarts)

        if base64_img is None:
            return None

        mime_type = "image/svg+xml" if cfg.format == ImageFormat.SVG else "image/png"
        return f"data:{mime_type};base64,{base64_img}"

    def render_grid(self,
                    smiles_list: List[str],
                    legends: Optional[List[str]] = None,
                    config: Optional[RenderConfig] = None) -> Optional[bytes]:
        """
        Render multiple molecules in a grid layout.

        Args:
            smiles_list: List of SMILES strings
            legends: Optional list of legend labels
            config: Optional config override

        Returns:
            Image bytes or None
        """
        if not self._is_ready:
            return None

        cfg = config or self.config

        try:
            mols = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    if cfg.compute_coords:
                        self._compute_2d_coords(mol, cfg.coord_gen_method)
                    mols.append(mol)
                else:
                    mols.append(None)

            # Filter out None values but track positions
            valid_mols = [m for m in mols if m is not None]
            valid_legends = None
            if legends:
                valid_legends = [l for m, l in zip(mols, legends) if m is not None]

            if not valid_mols:
                return None

            # Calculate grid dimensions
            n_mols = len(valid_mols)
            n_cols = min(cfg.mols_per_row, n_mols)
            n_rows = (n_mols + n_cols - 1) // n_cols

            # Single molecule dimensions
            mol_width = cfg.width // n_cols
            mol_height = cfg.height // n_rows

            # Highlight quat nitrogens in all molecules
            highlight_atoms_list = None
            if cfg.highlight_quat_nitrogen:
                highlight_atoms_list = []
                for mol in valid_mols:
                    quat_atoms = self._find_quat_nitrogens(mol)
                    highlight_atoms_list.append(quat_atoms)

            # Generate grid image
            if cfg.format == ImageFormat.SVG:
                img = Draw.MolsToGridImage(
                    valid_mols,
                    molsPerRow=n_cols,
                    subImgSize=(mol_width, mol_height),
                    legends=valid_legends,
                    highlightAtomLists=highlight_atoms_list,
                    useSVG=True
                )
                return img.encode('utf-8') if isinstance(img, str) else img
            else:
                img = Draw.MolsToGridImage(
                    valid_mols,
                    molsPerRow=n_cols,
                    subImgSize=(mol_width, mol_height),
                    legends=valid_legends,
                    highlightAtomLists=highlight_atoms_list
                )

                # Convert PIL Image to bytes
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                return img_buffer.getvalue()

        except Exception as e:
            logger.error(f"Grid rendering error: {e}")
            return None

    def render_grid_base64(self,
                           smiles_list: List[str],
                           legends: Optional[List[str]] = None,
                           config: Optional[RenderConfig] = None) -> Optional[str]:
        """Render grid and return as base64 encoded string."""
        img_bytes = self.render_grid(smiles_list, legends, config)
        if img_bytes is None:
            return None
        return base64.b64encode(img_bytes).decode('utf-8')

    def render_with_atom_map(self,
                             smiles: str,
                             atom_values: Dict[int, float],
                             config: Optional[RenderConfig] = None,
                             colormap: str = "RdYlGn") -> Optional[bytes]:
        """
        Render molecule with atoms colored by values (e.g., attention weights).

        Args:
            smiles: SMILES string
            atom_values: Dict mapping atom indices to values (0-1)
            config: Optional config override
            colormap: Matplotlib colormap name

        Returns:
            Image bytes or None
        """
        if not self._is_ready:
            return None

        cfg = config or self.config

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            if cfg.compute_coords:
                self._compute_2d_coords(mol, cfg.coord_gen_method)

            # Create color map
            try:
                import matplotlib.pyplot as plt
                cmap = plt.get_cmap(colormap)
            except ImportError:
                # Fallback to simple red-green gradient
                def cmap(val):
                    return (1 - val, val, 0)

            # Map values to colors
            highlight_atoms = list(atom_values.keys())
            highlight_colors = {}
            for atom_idx, value in atom_values.items():
                color = cmap(value)[:3]  # RGB only
                highlight_colors[atom_idx] = color

            # Draw
            if cfg.format == ImageFormat.SVG:
                drawer = rdMolDraw2D.MolDraw2DSVG(cfg.width, cfg.height)
            else:
                drawer = rdMolDraw2D.MolDraw2DCairo(cfg.width, cfg.height)

            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms,
                highlightAtomColors=highlight_colors
            )
            drawer.FinishDrawing()

            if cfg.format == ImageFormat.SVG:
                return drawer.GetDrawingText().encode('utf-8')
            else:
                return drawer.GetDrawingText()

        except Exception as e:
            logger.error(f"Atom map rendering error: {e}")
            return None

    def render_comparison(self,
                          smiles1: str,
                          smiles2: str,
                          legend1: str = "Molecule 1",
                          legend2: str = "Molecule 2",
                          config: Optional[RenderConfig] = None) -> Optional[bytes]:
        """
        Render two molecules side by side for comparison.

        Returns:
            Image bytes or None
        """
        cfg = config or RenderConfig(
            width=self.config.width * 2,
            height=self.config.height,
            mols_per_row=2
        )

        return self.render_grid([smiles1, smiles2], [legend1, legend2], cfg)

    def _compute_2d_coords(self, mol, method: str = "rdkit"):
        """Compute 2D coordinates for molecule"""
        try:
            if method == "coordgen":
                # Try CoordGen if available (better for complex structures)
                try:
                    from rdkit.Chem import rdCoordGen
                    rdCoordGen.AddCoords(mol)
                except ImportError:
                    AllChem.Compute2DCoords(mol)
            else:
                AllChem.Compute2DCoords(mol)
        except Exception:
            pass  # Keep existing coords if computation fails

    def _find_quat_nitrogens(self, mol) -> List[int]:
        """Find quaternary nitrogen atoms in molecule"""
        quat_atoms = []

        # SMARTS for quaternary nitrogen
        quat_patterns = [
            "[N+;X4;!$([N+](=O)=O)]",  # Aliphatic quat
            "[n+;X3]",                   # Aromatic (pyridinium)
        ]

        for smarts in quat_patterns:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    quat_atoms.extend(match)

        return list(set(quat_atoms))

    def get_molecule_info(self, smiles: str) -> Optional[Dict]:
        """
        Get molecule information useful for rendering/display.

        Returns:
            Dict with atom count, bond count, formula, etc.
        """
        if not self._is_ready:
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            quat_count = len(self._find_quat_nitrogens(mol))

            return {
                "num_atoms": mol.GetNumAtoms(),
                "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "formula": rdMolDescriptors.CalcMolFormula(mol),
                "has_quat_nitrogen": quat_count > 0,
                "quat_nitrogen_count": quat_count,
                "num_rings": rdMolDescriptors.CalcNumRings(mol),
                "is_valid": True
            }
        except Exception:
            return None

    def render_molecule_with_similarity(self,
                                        smiles: str,
                                        reference_smiles: str,
                                        config: Optional[RenderConfig] = None) -> Optional[bytes]:
        """
        Render molecule with atoms highlighted based on similarity to reference.

        Args:
            smiles: SMILES of molecule to render
            reference_smiles: SMILES of reference molecule
            config: Optional config override

        Returns:
            Image bytes or None
        """
        if not self._is_ready:
            return None

        cfg = config or self.config

        try:
            mol = Chem.MolFromSmiles(smiles)
            ref_mol = Chem.MolFromSmiles(reference_smiles)

            if mol is None or ref_mol is None:
                return None

            if cfg.compute_coords:
                self._compute_2d_coords(mol, cfg.coord_gen_method)

            # Find maximum common substructure
            from rdkit.Chem import rdFMCS
            mcs_result = rdFMCS.FindMCS([mol, ref_mol], timeout=1)

            highlight_atoms = []
            if mcs_result.smartsString:
                mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
                if mcs_mol:
                    matches = mol.GetSubstructMatches(mcs_mol)
                    if matches:
                        highlight_atoms = list(matches[0])

            # Draw with MCS highlighted
            if cfg.format == ImageFormat.SVG:
                drawer = rdMolDraw2D.MolDraw2DSVG(cfg.width, cfg.height)
            else:
                drawer = rdMolDraw2D.MolDraw2DCairo(cfg.width, cfg.height)

            if highlight_atoms:
                drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
            else:
                drawer.DrawMolecule(mol)

            drawer.FinishDrawing()

            if cfg.format == ImageFormat.SVG:
                return drawer.GetDrawingText().encode('utf-8')
            else:
                return drawer.GetDrawingText()

        except Exception as e:
            logger.error(f"Similarity rendering error: {e}")
            return None
