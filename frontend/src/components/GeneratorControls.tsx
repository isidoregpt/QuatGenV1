import React, { useState } from 'react';
import { Play, Square, Settings } from 'lucide-react';
import { useStore } from '../hooks/useStore';
import { useBackend } from '../hooks/useBackend';

export function GeneratorControls() {
  const { weights, setWeights, constraints, setConstraints, generationStatus } = useStore();
  const { startGeneration, stopGeneration } = useBackend();
  const [numMolecules, setNumMolecules] = useState(1000);
  const [batchSize, setBatchSize] = useState(64);
  const isRunning = generationStatus?.is_running ?? false;

  const handleStart = async () => {
    await startGeneration({ num_molecules: numMolecules, constraints, weights, batch_size: batchSize, use_gpu: true, num_workers: 8 });
  };

  return (
    <div className="p-4 space-y-6">
      <div>
        <h2 className="text-lg font-semibold mb-4">Generation</h2>
        <div className="flex gap-2 mb-4">
          {!isRunning ? (
            <button onClick={handleStart} className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg font-medium">
              <Play size={18} /> Start
            </button>
          ) : (
            <button onClick={stopGeneration} className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-500 rounded-lg font-medium">
              <Square size={18} /> Stop
            </button>
          )}
        </div>
        <div className="mb-4">
          <label className="block text-sm text-gray-400 mb-1">Target Molecules</label>
          <input type="number" value={numMolecules} onChange={(e) => setNumMolecules(Number(e.target.value))}
            className="w-full px-3 py-2 bg-gray-800 rounded border border-gray-700" min={1} max={100000} />
        </div>
      </div>
      <div>
        <h3 className="font-medium mb-3 flex items-center gap-2"><Settings size={16} /> Objective Weights</h3>
        {(['efficacy', 'safety', 'environmental', 'sa_score'] as const).map((key) => (
          <div key={key} className="mb-3">
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-400 capitalize">{key === 'sa_score' ? 'SA Score' : key}</span>
              <span>{(weights[key] * 100).toFixed(0)}%</span>
            </div>
            <input type="range" min="0" max="100" value={weights[key] * 100}
              onChange={(e) => setWeights({ ...weights, [key]: Number(e.target.value) / 100 })}
              className="w-full accent-emerald-500" />
          </div>
        ))}
      </div>
      <div>
        <h3 className="font-medium mb-3">Constraints</h3>
        <div className="grid grid-cols-2 gap-3">
          <div><label className="block text-xs text-gray-400 mb-1">Min MW</label>
            <input type="number" value={constraints.min_mw} onChange={(e) => setConstraints({ min_mw: Number(e.target.value) })}
              className="w-full px-2 py-1 bg-gray-800 rounded border border-gray-700 text-sm" /></div>
          <div><label className="block text-xs text-gray-400 mb-1">Max MW</label>
            <input type="number" value={constraints.max_mw} onChange={(e) => setConstraints({ max_mw: Number(e.target.value) })}
              className="w-full px-2 py-1 bg-gray-800 rounded border border-gray-700 text-sm" /></div>
        </div>
      </div>
    </div>
  );
}
