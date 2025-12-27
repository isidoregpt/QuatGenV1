import React from 'react';
import { X, Copy, Star, ExternalLink } from 'lucide-react';
import { useStore } from '../hooks/useStore';
import type { Molecule } from '../types';

export function MoleculeDetail({ molecule }: { molecule: Molecule }) {
  const { setSelectedMolecule } = useStore();
  const copySmiles = () => navigator.clipboard.writeText(molecule.smiles);

  return (
    <div className="p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">Molecule #{molecule.id}</h3>
        <button onClick={() => setSelectedMolecule(null)} className="p-1 hover:bg-gray-700 rounded"><X size={18} /></button>
      </div>
      <div className="mb-4">
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm text-gray-400">SMILES</span>
          <button onClick={copySmiles} className="p-1 hover:bg-gray-700 rounded text-gray-400"><Copy size={14} /></button>
        </div>
        <code className="block p-2 bg-gray-800 rounded text-xs font-mono break-all">{molecule.smiles}</code>
      </div>
      <div className="mb-4">
        <h4 className="text-sm text-gray-400 mb-2">Scores</h4>
        {['efficacy_score', 'safety_score', 'environmental_score', 'sa_score'].map((key) => (
          <div key={key} className="mb-2">
            <div className="flex justify-between text-sm mb-1">
              <span className="capitalize">{key.replace('_score', '')}</span>
              <span>{(molecule[key as keyof Molecule] as number).toFixed(1)}</span>
            </div>
            <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
              <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${molecule[key as keyof Molecule]}%` }} />
            </div>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-2 text-sm mb-4">
        <div className="bg-gray-800 rounded p-2"><div className="text-gray-400 text-xs">MW</div><div>{molecule.molecular_weight?.toFixed(1) ?? '-'}</div></div>
        <div className="bg-gray-800 rounded p-2"><div className="text-gray-400 text-xs">LogP</div><div>{molecule.logp?.toFixed(2) ?? '-'}</div></div>
      </div>
      <div className="flex gap-2">
        {molecule.is_pareto && <span className="px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded text-xs">Pareto</span>}
        {molecule.is_starred && <span className="px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded text-xs flex items-center gap-1"><Star size={12} fill="currentColor" /> Starred</span>}
      </div>
      <div className="mt-4 pt-4 border-t border-gray-700">
        <a href={`https://pubchem.ncbi.nlm.nih.gov/#query=${encodeURIComponent(molecule.smiles)}`} target="_blank" rel="noopener noreferrer"
          className="flex items-center gap-2 text-sm text-emerald-400 hover:text-emerald-300"><ExternalLink size={14} /> Search in PubChem</a>
      </div>
    </div>
  );
}
