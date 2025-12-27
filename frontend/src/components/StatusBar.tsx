import React from 'react';
import { Wifi, WifiOff, Cpu } from 'lucide-react';
import { useStore } from '../hooks/useStore';

export function StatusBar() {
  const { isConnected, generationStatus } = useStore();
  const formatTime = (s: number) => s < 60 ? `${s.toFixed(0)}s` : s < 3600 ? `${(s / 60).toFixed(0)}m` : `${(s / 3600).toFixed(1)}h`;

  return (
    <div className="flex items-center gap-4 px-4 py-2 bg-gray-800 border-t border-gray-700 text-sm">
      <div className="flex items-center gap-1.5">
        {isConnected ? (<><Wifi size={14} className="text-emerald-400" /><span className="text-emerald-400">Connected</span></>) :
                       (<><WifiOff size={14} className="text-red-400" /><span className="text-red-400">Disconnected</span></>)}
      </div>
      <div className="w-px h-4 bg-gray-600" />
      {generationStatus?.is_running && (<>
        <div className="flex items-center gap-1.5"><Cpu size={14} className="text-emerald-400 animate-pulse" /><span>Generating...</span></div>
        <span className="text-gray-400">{generationStatus.molecules_generated} molecules</span>
        <span className="text-gray-400">{generationStatus.molecules_per_hour.toFixed(0)}/hr</span>
        <span className="text-gray-400">Pareto: {generationStatus.pareto_frontier_size}</span>
        <span className="text-gray-400">ETA: {formatTime(generationStatus.estimated_remaining_seconds)}</span>
      </>)}
      <div className="flex-1" />
      {generationStatus && (
        <div className="flex items-center gap-3 text-xs">
          <span className="text-gray-400">Best:</span>
          <span>Eff {generationStatus.top_scores.efficacy.toFixed(0)}</span>
          <span>Safe {generationStatus.top_scores.safety.toFixed(0)}</span>
          <span>Env {generationStatus.top_scores.environmental.toFixed(0)}</span>
        </div>
      )}
    </div>
  );
}
