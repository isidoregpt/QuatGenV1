import React from 'react';
import { X, Copy, Star, ExternalLink } from 'lucide-react';
import { useStore } from '../hooks/useStore';
import type { Molecule } from '../types';

interface Props {
  molecule: Molecule;
}

export function MoleculeDetail({ molecule }: Props) {
  const { setSelectedMolecule } = useStore();

  const copySmiles = () => {
    navigator.clipboard.writeText(molecule.smiles);
  };

  return (
    <div className="p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">Molecule #{molecule.id}</h3>
        <button
          onClick={() => setSelectedMolecule(null)}
          className="p-1 hover:bg-gray-700 rounded"
        >
          <X size={18} />
        </button>
      </div>

      {/* SMILES */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm text-gray-400">SMILES</span>
          <button
            onClick={copySmiles}
            className="p-1 hover:bg-gray-700 rounded text-gray-400 hover:text-white"
            title="Copy SMILES"
          >
            <Copy size={14} />
          </button>
        </div>
        <code className="block p-2 bg-gray-800 rounded text-xs font-mono break-all">
          {molecule.smiles}
        </code>
      </div>

      {/* Scores */}
      <div className="mb-4">
        <h4 className="text-sm text-gray-400 mb-2">Scores</h4>
        <div className="space-y-2">
          <ScoreBar label="Efficacy" value={molecule.efficacy_score} color="emerald" />
          <ScoreBar label="Safety" value={molecule.safety_score} color="blue" />
          <ScoreBar label="Environmental" value={molecule.environmental_score} color="green" />
          <ScoreBar label="SA Score" value={molecule.sa_score} color="purple" />
        </div>
      </div>

      {/* Properties */}
      <div className="mb-4">
        <h4 className="text-sm text-gray-400 mb-2">Properties</h4>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="bg-gray-800 rounded p-2">
            <div className="text-gray-400 text-xs">Molecular Weight</div>
            <div className="font-medium">{molecule.molecular_weight?.toFixed(1) ?? '-'}</div>
          </div>
          <div className="bg-gray-800 rounded p-2">
            <div className="text-gray-400 text-xs">LogP</div>
            <div className="font-medium">{molecule.logp?.toFixed(2) ?? '-'}</div>
          </div>
          <div className="bg-gray-800 rounded p-2">
            <div className="text-gray-400 text-xs">Chain Length</div>
            <div className="font-medium">{molecule.chain_length ?? '-'}</div>
          </div>
          <div className="bg-gray-800 rounded p-2">
            <div className="text-gray-400 text-xs">Valid Quat</div>
            <div className="font-medium">{molecule.is_valid_quat ? 'Yes' : 'No'}</div>
          </div>
        </div>
      </div>

      {/* Flags */}
      <div className="flex gap-2">
        {molecule.is_pareto && (
          <span className="px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded text-xs">
            Pareto Optimal
          </span>
        )}
        {molecule.is_starred && (
          <span className="px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded text-xs flex items-center gap-1">
            <Star size={12} fill="currentColor" />
            Starred
          </span>
        )}
      </div>

      {/* Actions */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <a
          href={`https://pubchem.ncbi.nlm.nih.gov/#query=${encodeURIComponent(molecule.smiles)}`}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-sm text-emerald-400 hover:text-emerald-300"
        >
          <ExternalLink size={14} />
          Search in PubChem
        </a>
      </div>
    </div>
  );
}

function ScoreBar({ 
  label, 
  value, 
  color 
}: { 
  label: string; 
  value: number; 
  color: string;
}) {
  const colorClasses: Record<string, string> = {
    emerald: 'bg-emerald-500',
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    purple: 'bg-purple-500',
  };

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span>{label}</span>
        <span className="font-medium">{value.toFixed(1)}</span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${colorClasses[color]} rounded-full transition-all`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}
