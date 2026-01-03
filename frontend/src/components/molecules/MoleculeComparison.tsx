/**
 * MoleculeComparison - Side-by-side comparison of two molecules
 */

import React, { useState, useEffect } from 'react';
import { MoleculeImage } from './MoleculeImage';
import { moleculeApi } from '../../services/api';

interface Molecule {
  id?: number;
  smiles: string;
  efficacy_score?: number;
  safety_score?: number;
  environmental_score?: number;
  sa_score?: number;
  combined_score?: number;
  name?: string;
}

interface MoleculeComparisonProps {
  molecule1: Molecule;
  molecule2: Molecule;
  title?: string;
  onClose?: () => void;
}

export const MoleculeComparison: React.FC<MoleculeComparisonProps> = ({
  molecule1,
  molecule2,
  title = 'Molecule Comparison',
  onClose
}) => {
  const [comparisonImage, setComparisonImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadComparison = async () => {
      try {
        const response = await moleculeApi.renderComparison(
          molecule1.smiles,
          molecule2.smiles,
          molecule1.name || `#${molecule1.id || '1'}`,
          molecule2.name || `#${molecule2.id || '2'}`
        );
        setComparisonImage(`data:${response.mime_type};base64,${response.image_base64}`);
      } catch (err) {
        console.error('Failed to load comparison:', err);
      } finally {
        setLoading(false);
      }
    };

    loadComparison();
  }, [molecule1.smiles, molecule2.smiles, molecule1.id, molecule2.id, molecule1.name, molecule2.name]);

  const compareValue = (val1?: number, val2?: number) => {
    if (val1 === undefined || val2 === undefined) return 'neutral';
    if (val1 > val2 + 5) return 'better';
    if (val1 < val2 - 5) return 'worse';
    return 'similar';
  };

  const comparisonClass = (comparison: string) => {
    switch (comparison) {
      case 'better': return 'text-emerald-400 font-semibold';
      case 'worse': return 'text-red-400';
      case 'similar': return 'text-gray-300';
      default: return 'text-gray-500';
    }
  };

  const properties: { key: keyof Molecule; label: string }[] = [
    { key: 'efficacy_score', label: 'Efficacy' },
    { key: 'safety_score', label: 'Safety' },
    { key: 'environmental_score', label: 'Environmental' },
    { key: 'sa_score', label: 'Synthesis' },
    { key: 'combined_score', label: 'Combined' }
  ];

  return (
    <div className="molecule-comparison bg-gray-800 rounded-lg shadow-xl p-6 border border-gray-700">
      {/* Header */}
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-100">{title}</h3>
        {onClose && (
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200 transition-colors"
          >
            ✕
          </button>
        )}
      </div>

      {/* Structure Images */}
      <div className="comparison-images mb-6">
        {loading ? (
          <div className="flex items-center justify-center py-8 bg-gray-900 rounded-lg">
            <div className="loading-spinner mr-3" />
            <span className="text-gray-400">Loading comparison...</span>
          </div>
        ) : comparisonImage ? (
          <img
            src={comparisonImage}
            alt="Molecule comparison"
            className="w-full rounded-lg bg-white"
          />
        ) : (
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-900 rounded-lg p-2">
              <MoleculeImage smiles={molecule1.smiles} width={350} height={250} />
              <p className="text-center text-sm text-gray-400 mt-2">
                {molecule1.name || `Molecule #${molecule1.id || '1'}`}
              </p>
            </div>
            <div className="bg-gray-900 rounded-lg p-2">
              <MoleculeImage smiles={molecule2.smiles} width={350} height={250} />
              <p className="text-center text-sm text-gray-400 mt-2">
                {molecule2.name || `Molecule #${molecule2.id || '2'}`}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Property Comparison Table */}
      <table className="w-full">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="text-left py-2 text-gray-400 font-medium">Property</th>
            <th className="text-center py-2 text-gray-400 font-medium">
              {molecule1.name || `#${molecule1.id || '1'}`}
            </th>
            <th className="text-center py-2 text-gray-400 font-medium">
              {molecule2.name || `#${molecule2.id || '2'}`}
            </th>
            <th className="text-center py-2 text-gray-400 font-medium">Δ</th>
          </tr>
        </thead>
        <tbody>
          {properties.map(({ key, label }) => {
            const val1 = molecule1[key] as number | undefined;
            const val2 = molecule2[key] as number | undefined;
            const comparison = compareValue(val1, val2);
            const diff = val1 !== undefined && val2 !== undefined
              ? (val1 - val2).toFixed(1)
              : '-';

            return (
              <tr key={key} className="border-b border-gray-700/50">
                <td className="py-3 text-gray-300">{label}</td>
                <td className={`text-center py-3 ${comparisonClass(comparison)}`}>
                  {val1?.toFixed(1) ?? '-'}
                </td>
                <td className={`text-center py-3 ${comparisonClass(
                  comparison === 'better' ? 'worse' : comparison === 'worse' ? 'better' : 'similar'
                )}`}>
                  {val2?.toFixed(1) ?? '-'}
                </td>
                <td className={`text-center py-3 ${
                  Number(diff) > 0 ? 'text-emerald-400' : Number(diff) < 0 ? 'text-red-400' : 'text-gray-400'
                }`}>
                  {Number(diff) > 0 ? '+' : ''}{diff}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default MoleculeComparison;
