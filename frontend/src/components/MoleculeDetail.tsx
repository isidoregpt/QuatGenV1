import React, { useState } from 'react';
import { X, Copy, Star, ExternalLink } from 'lucide-react';
import { useStore } from '../hooks/useStore';
import { MoleculeImage } from './molecules/MoleculeImage';
import type { Molecule } from '../types';

type TabType = 'structure' | 'scores' | 'properties';

export function MoleculeDetail({ molecule }: { molecule: Molecule }) {
  const { setSelectedMolecule } = useStore();
  const [activeTab, setActiveTab] = useState<TabType>('structure');
  const copySmiles = () => navigator.clipboard.writeText(molecule.smiles);

  const scoreBar = (label: string, score: number, colorClass: string) => (
    <div className="mb-3">
      <div className="flex justify-between text-sm mb-1">
        <span className="capitalize text-gray-400">{label}</span>
        <span className="font-medium">{score?.toFixed(1)}</span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${colorClass}`}
          style={{ width: `${Math.min(100, score)}%` }}
        />
      </div>
    </div>
  );

  return (
    <div className="p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-lg">Molecule #{molecule.id}</h3>
        <button
          onClick={() => setSelectedMolecule(null)}
          className="p-1 hover:bg-gray-700 rounded transition-colors"
        >
          <X size={18} />
        </button>
      </div>

      {/* Status Badges */}
      <div className="flex gap-2 mb-4">
        {molecule.is_pareto && (
          <span className="px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded text-xs font-medium">
            Pareto
          </span>
        )}
        {molecule.is_starred && (
          <span className="px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded text-xs flex items-center gap-1">
            <Star size={12} fill="currentColor" /> Starred
          </span>
        )}
        {molecule.is_valid_quat && (
          <span className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded text-xs font-medium">
            Valid Quat
          </span>
        )}
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-700 mb-4">
        {(['structure', 'scores', 'properties'] as TabType[]).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 capitalize text-sm font-medium transition-colors ${
              activeTab === tab
                ? 'border-b-2 border-emerald-500 text-emerald-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'structure' && (
        <div>
          {/* 2D Structure Image */}
          <div className="mb-4 bg-gray-900 rounded-lg overflow-hidden">
            <MoleculeImage
              moleculeId={molecule.id}
              smiles={molecule.smiles}
              width={350}
              height={280}
              format="svg"
              highlightQuat={true}
              className="w-full"
            />
          </div>

          {/* SMILES */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm text-gray-400">SMILES</span>
              <button
                onClick={copySmiles}
                className="p-1 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200 transition-colors"
                title="Copy SMILES"
              >
                <Copy size={14} />
              </button>
            </div>
            <code className="block p-2 bg-gray-800 rounded text-xs font-mono break-all text-gray-300">
              {molecule.smiles}
            </code>
          </div>
        </div>
      )}

      {activeTab === 'scores' && (
        <div>
          {scoreBar('efficacy', molecule.efficacy_score, 'bg-blue-500')}
          {scoreBar('safety', molecule.safety_score, 'bg-emerald-500')}
          {scoreBar('environmental', molecule.environmental_score, 'bg-teal-500')}
          {scoreBar('synthesis', molecule.sa_score, 'bg-purple-500')}

          <div className="mt-4 pt-4 border-t border-gray-700">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Combined Score</span>
              <span className="text-2xl font-bold text-emerald-400">
                {molecule.combined_score?.toFixed(1)}
              </span>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'properties' && (
        <div>
          <div className="grid grid-cols-2 gap-2 text-sm mb-4">
            <div className="bg-gray-800 rounded p-3">
              <div className="text-gray-400 text-xs mb-1">Molecular Weight</div>
              <div className="font-medium">{molecule.molecular_weight?.toFixed(1) ?? '-'} Da</div>
            </div>
            <div className="bg-gray-800 rounded p-3">
              <div className="text-gray-400 text-xs mb-1">LogP</div>
              <div className="font-medium">{molecule.logp?.toFixed(2) ?? '-'}</div>
            </div>
            <div className="bg-gray-800 rounded p-3">
              <div className="text-gray-400 text-xs mb-1">Chain Length</div>
              <div className="font-medium">{molecule.chain_length ?? '-'}</div>
            </div>
            <div className="bg-gray-800 rounded p-3">
              <div className="text-gray-400 text-xs mb-1">Created</div>
              <div className="font-medium">
                {molecule.created_at
                  ? new Date(molecule.created_at).toLocaleDateString()
                  : '-'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* External Links */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <a
          href={`https://pubchem.ncbi.nlm.nih.gov/#query=${encodeURIComponent(molecule.smiles)}`}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-sm text-emerald-400 hover:text-emerald-300 transition-colors"
        >
          <ExternalLink size={14} /> Search in PubChem
        </a>
      </div>
    </div>
  );
}
