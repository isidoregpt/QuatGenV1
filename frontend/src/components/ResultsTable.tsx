import React, { useEffect } from 'react';
import { Star, Filter } from 'lucide-react';
import { useStore } from '../hooks/useStore';
import { useBackend } from '../hooks/useBackend';
import type { Molecule } from '../types';

function ScoreBadge({ score }: { score: number }) {
  const colorClass = score >= 70 ? 'score-high' : score >= 40 ? 'score-medium' : 'score-low';
  return <span className={`px-2 py-0.5 rounded text-xs font-medium ${colorClass}`}>{score.toFixed(0)}</span>;
}

export function ResultsTable() {
  const { molecules, setSelectedMolecule, selectedMolecule, filters, setFilters, totalMolecules } = useStore();
  const { fetchMolecules, toggleStar } = useBackend();

  useEffect(() => { fetchMolecules(); }, [filters]);

  const handleStarClick = async (e: React.MouseEvent, mol: Molecule) => {
    e.stopPropagation();
    await toggleStar(mol.id, !mol.is_starred);
    fetchMolecules();
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-4 px-4 py-2 bg-gray-800 border-b border-gray-700">
        <Filter size={16} className="text-gray-400" />
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={filters.pareto_only} onChange={(e) => setFilters({ pareto_only: e.target.checked })} className="accent-emerald-500" />
          <span className="text-sm">Pareto Only</span>
        </label>
        <label className="flex items-center gap-1.5 ml-3">
          <input type="checkbox" checked={filters.starred_only} onChange={(e) => setFilters({ starred_only: e.target.checked })} className="accent-emerald-500" />
          <span className="text-sm">Starred</span>
        </label>
        <div className="flex-1" />
        <span className="text-sm text-gray-400">{molecules.length} / {totalMolecules} molecules</span>
      </div>
      <div className="flex-1 overflow-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-800 sticky top-0">
            <tr>
              <th className="px-3 py-2 text-left w-8"></th>
              <th className="px-3 py-2 text-left">SMILES</th>
              <th className="px-3 py-2 text-center w-16">Eff</th>
              <th className="px-3 py-2 text-center w-16">Safe</th>
              <th className="px-3 py-2 text-center w-16">Env</th>
              <th className="px-3 py-2 text-center w-16">SA</th>
              <th className="px-3 py-2 text-center w-12">P</th>
            </tr>
          </thead>
          <tbody>
            {molecules.map((mol) => (
              <tr key={mol.id} onClick={() => setSelectedMolecule(mol)}
                className={`border-b border-gray-800 cursor-pointer ${selectedMolecule?.id === mol.id ? 'bg-emerald-900/30' : 'hover:bg-gray-800/50'}`}>
                <td className="px-3 py-2">
                  <button onClick={(e) => handleStarClick(e, mol)} className={mol.is_starred ? 'text-yellow-400' : 'text-gray-600 hover:text-gray-400'}>
                    <Star size={14} fill={mol.is_starred ? 'currentColor' : 'none'} />
                  </button>
                </td>
                <td className="px-3 py-2"><code className="text-xs font-mono">{mol.smiles.slice(0, 50)}{mol.smiles.length > 50 ? '...' : ''}</code></td>
                <td className="px-3 py-2 text-center"><ScoreBadge score={mol.efficacy_score} /></td>
                <td className="px-3 py-2 text-center"><ScoreBadge score={mol.safety_score} /></td>
                <td className="px-3 py-2 text-center"><ScoreBadge score={mol.environmental_score} /></td>
                <td className="px-3 py-2 text-center"><ScoreBadge score={mol.sa_score} /></td>
                <td className="px-3 py-2 text-center">{mol.is_pareto && <span className="text-emerald-400">●</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
