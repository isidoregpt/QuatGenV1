/**
 * MoleculeGrid - Grid display of multiple molecules with images
 */

import React from 'react';
import { MoleculeCard } from './MoleculeCard';

interface Molecule {
  id: number;
  smiles: string;
  efficacy_score: number;
  safety_score: number;
  environmental_score: number;
  sa_score: number;
  combined_score: number;
}

interface MoleculeGridProps {
  molecules: Molecule[];
  loading?: boolean;
  onSelectMolecule?: (id: number) => void;
  onViewDetails?: (id: number) => void;
  selectedIds?: number[];
  columns?: 2 | 3 | 4;
  compact?: boolean;
}

export const MoleculeGrid: React.FC<MoleculeGridProps> = ({
  molecules,
  loading = false,
  onSelectMolecule,
  onViewDetails,
  selectedIds = [],
  columns = 3,
  compact = false
}) => {
  const gridCols = {
    2: 'grid-cols-2',
    3: 'grid-cols-3',
    4: 'grid-cols-4'
  };

  if (loading) {
    return (
      <div className="molecule-grid-loading flex items-center justify-center py-12">
        <div className="loading-spinner mr-3" />
        <span className="text-gray-400">Loading molecules...</span>
      </div>
    );
  }

  if (molecules.length === 0) {
    return (
      <div className="molecule-grid-empty flex flex-col items-center justify-center py-12">
        <span className="text-4xl mb-3">🧪</span>
        <span className="text-gray-400 text-lg">No molecules to display</span>
        <span className="text-sm text-gray-500 mt-1">
          Generate some molecules to see them here
        </span>
      </div>
    );
  }

  return (
    <div className={`molecule-grid grid ${gridCols[columns]} gap-4 p-4`}>
      {molecules.map(molecule => (
        <MoleculeCard
          key={molecule.id}
          molecule={molecule}
          selected={selectedIds.includes(molecule.id)}
          onSelect={onSelectMolecule}
          onViewDetails={onViewDetails}
          compact={compact}
        />
      ))}
    </div>
  );
};

export default MoleculeGrid;
