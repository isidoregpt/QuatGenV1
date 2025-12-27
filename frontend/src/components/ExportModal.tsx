import React, { useState } from 'react';
import { X, Download, FileText, Database, File } from 'lucide-react';
import { useStore } from '../hooks/useStore';
import { useBackend } from '../hooks/useBackend';
import type { ExportFormat } from '../types';

interface Props {
  onClose: () => void;
}

export function ExportModal({ onClose }: Props) {
  const { filters, selectedIds } = useStore();
  const { exportMolecules } = useBackend();
  
  const [format, setFormat] = useState<ExportFormat>('csv');
  const [includeStructures, setIncludeStructures] = useState(true);
  const [includeProperties, setIncludeProperties] = useState(true);
  const [includeScores, setIncludeScores] = useState(true);
  const [useSelection, setUseSelection] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = async () => {
    setIsExporting(true);
    try {
      await exportMolecules({
        molecule_ids: useSelection && selectedIds.size > 0 
          ? Array.from(selectedIds) 
          : undefined,
        format,
        include_structures: includeStructures,
        include_properties: includeProperties,
        include_scores: includeScores,
        pareto_only: filters.pareto_only,
        starred_only: filters.starred_only,
      });
      onClose();
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsExporting(false);
    }
  };

  const formatOptions = [
    { id: 'csv', label: 'CSV', icon: FileText, desc: 'Tabular data for Excel/Sheets' },
    { id: 'sdf', label: 'SDF', icon: Database, desc: 'Chemistry format with structures' },
    { id: 'pdf', label: 'PDF', icon: File, desc: 'Report with visualizations' },
  ] as const;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg w-full max-w-md mx-4">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <h2 className="font-semibold">Export Molecules</h2>
          <button onClick={onClose} className="p-1 hover:bg-gray-700 rounded">
            <X size={18} />
          </button>
        </div>

        <div className="p-4 space-y-4">
          {/* Format selection */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">Format</label>
            <div className="grid grid-cols-3 gap-2">
              {formatOptions.map((opt) => (
                <button
                  key={opt.id}
                  onClick={() => setFormat(opt.id)}
                  className={`
                    p-3 rounded-lg border text-center transition
                    ${format === opt.id 
                      ? 'border-emerald-500 bg-emerald-500/10' 
                      : 'border-gray-600 hover:border-gray-500'}
                  `}
                >
                  <opt.icon size={20} className="mx-auto mb-1" />
                  <div className="text-sm font-medium">{opt.label}</div>
                  <div className="text-xs text-gray-400">{opt.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Options */}
          <div className="space-y-2">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={includeScores}
                onChange={(e) => setIncludeScores(e.target.checked)}
                className="accent-emerald-500"
              />
              <span className="text-sm">Include scores</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={includeProperties}
                onChange={(e) => setIncludeProperties(e.target.checked)}
                className="accent-emerald-500"
              />
              <span className="text-sm">Include properties</span>
            </label>
            {format !== 'csv' && (
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={includeStructures}
                  onChange={(e) => setIncludeStructures(e.target.checked)}
                  className="accent-emerald-500"
                />
                <span className="text-sm">Include 2D structures</span>
              </label>
            )}
            {selectedIds.size > 0 && (
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={useSelection}
                  onChange={(e) => setUseSelection(e.target.checked)}
                  className="accent-emerald-500"
                />
                <span className="text-sm">
                  Export selected only ({selectedIds.size} molecules)
                </span>
              </label>
            )}
          </div>
        </div>

        <div className="flex justify-end gap-2 px-4 py-3 border-t border-gray-700">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-400 hover:text-white"
          >
            Cancel
          </button>
          <button
            onClick={handleExport}
            disabled={isExporting}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg font-medium disabled:opacity-50"
          >
            <Download size={16} />
            {isExporting ? 'Exporting...' : 'Export'}
          </button>
        </div>
      </div>
    </div>
  );
}
