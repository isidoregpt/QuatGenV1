/**
 * MoleculeCard - Card component displaying molecule with structure image
 */

import React, { useState } from 'react';
import { MoleculeImage } from './MoleculeImage';

interface Molecule {
  id: number;
  smiles: string;
  efficacy_score: number;
  safety_score: number;
  environmental_score: number;
  sa_score: number;
  combined_score: number;
  molecular_weight?: number;
  logp?: number;
}

interface MoleculeCardProps {
  molecule: Molecule;
  selected?: boolean;
  onSelect?: (id: number) => void;
  onViewDetails?: (id: number) => void;
  compact?: boolean;
}

export const MoleculeCard: React.FC<MoleculeCardProps> = ({
  molecule,
  selected = false,
  onSelect,
  onViewDetails,
  compact = false
}) => {
  const [imageLoaded, setImageLoaded] = useState(false);

  const scoreColor = (score: number) => {
    if (score >= 70) return 'text-emerald-400';
    if (score >= 50) return 'text-yellow-400';
    return 'text-red-400';
  };

  const scoreBgColor = (score: number) => {
    if (score >= 70) return 'bg-emerald-500/20';
    if (score >= 50) return 'bg-yellow-500/20';
    return 'bg-red-500/20';
  };

  return (
    <div
      className={`
        molecule-card bg-gray-800 rounded-lg overflow-hidden border
        ${selected ? 'border-emerald-500 ring-2 ring-emerald-500/30' : 'border-gray-700'}
        ${compact ? 'p-2' : 'p-4'}
        transition-all hover:border-gray-600 hover:shadow-lg
      `}
    >
      {/* Structure Image */}
      <div className="molecule-image-container mb-3 bg-gray-900 rounded-lg overflow-hidden">
        <MoleculeImage
          moleculeId={molecule.id}
          smiles={molecule.smiles}
          width={compact ? 150 : 250}
          height={compact ? 100 : 180}
          highlightQuat={true}
          onLoad={() => setImageLoaded(true)}
          onClick={() => onViewDetails?.(molecule.id)}
        />
      </div>

      {/* Scores */}
      <div className={`scores-grid grid ${compact ? 'grid-cols-2 gap-1' : 'grid-cols-2 gap-2'}`}>
        <div className={`score-item p-2 rounded ${scoreBgColor(molecule.efficacy_score)}`}>
          <span className="score-label text-xs text-gray-400 block">Efficacy</span>
          <span className={`score-value font-semibold ${scoreColor(molecule.efficacy_score)}`}>
            {molecule.efficacy_score?.toFixed(1) ?? '-'}
          </span>
        </div>
        <div className={`score-item p-2 rounded ${scoreBgColor(molecule.safety_score)}`}>
          <span className="score-label text-xs text-gray-400 block">Safety</span>
          <span className={`score-value font-semibold ${scoreColor(molecule.safety_score)}`}>
            {molecule.safety_score?.toFixed(1) ?? '-'}
          </span>
        </div>
        {!compact && (
          <>
            <div className={`score-item p-2 rounded ${scoreBgColor(molecule.environmental_score)}`}>
              <span className="score-label text-xs text-gray-400 block">Env</span>
              <span className={`score-value font-semibold ${scoreColor(molecule.environmental_score)}`}>
                {molecule.environmental_score?.toFixed(1) ?? '-'}
              </span>
            </div>
            <div className={`score-item p-2 rounded ${scoreBgColor(molecule.sa_score)}`}>
              <span className="score-label text-xs text-gray-400 block">Synth</span>
              <span className={`score-value font-semibold ${scoreColor(molecule.sa_score)}`}>
                {molecule.sa_score?.toFixed(1) ?? '-'}
              </span>
            </div>
          </>
        )}
      </div>

      {/* Combined Score */}
      <div className="combined-score mt-3 pt-3 border-t border-gray-700">
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-gray-400">Combined</span>
          <span className={`text-lg font-bold ${scoreColor(molecule.combined_score)}`}>
            {molecule.combined_score?.toFixed(1) ?? '-'}
          </span>
        </div>
      </div>

      {/* Actions */}
      <div className="actions mt-3 flex gap-2">
        {onSelect && (
          <button
            onClick={() => onSelect(molecule.id)}
            className={`
              flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors
              ${selected
                ? 'bg-emerald-500 text-white hover:bg-emerald-600'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}
            `}
          >
            {selected ? 'Selected' : 'Select'}
          </button>
        )}
        {onViewDetails && (
          <button
            onClick={() => onViewDetails(molecule.id)}
            className="flex-1 px-3 py-1.5 rounded text-sm font-medium bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
          >
            Details
          </button>
        )}
      </div>
    </div>
  );
};

export default MoleculeCard;
