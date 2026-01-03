/**
 * BenchmarkPanel - Main benchmark interface
 */

import React, { useState, useEffect } from 'react';
import { BarChart3, AlertCircle } from 'lucide-react';
import { benchmarkApi, BenchmarkResult, BenchmarkReportSummary, ReferenceCompound } from '../../services/api';
import { BenchmarkResultView } from './BenchmarkResultView';
import { BenchmarkReport } from './BenchmarkReport';
import { ReferenceCompoundCard } from './ReferenceCompoundCard';

type BenchmarkView = 'single' | 'batch' | 'report' | 'references';

interface BenchmarkPanelProps {
  selectedMoleculeId?: number;
  onSelectMolecule?: (id: number) => void;
  onClose?: () => void;
}

export const BenchmarkPanel: React.FC<BenchmarkPanelProps> = ({
  selectedMoleculeId,
  onSelectMolecule,
  onClose
}) => {
  const [view, setView] = useState<BenchmarkView>('single');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Single benchmark state
  const [smilesInput, setSmilesInput] = useState('');
  const [singleResult, setSingleResult] = useState<BenchmarkResult | null>(null);

  // Batch benchmark state
  const [batchResults, setBatchResults] = useState<BenchmarkResult[]>([]);
  const [topN, setTopN] = useState(20);
  const [minScore, setMinScore] = useState(50);

  // Report state
  const [report, setReport] = useState<BenchmarkReportSummary | null>(null);

  // References state
  const [references, setReferences] = useState<ReferenceCompound[]>([]);

  // Service status
  const [serviceAvailable, setServiceAvailable] = useState<boolean | null>(null);
  const [referenceCount, setReferenceCount] = useState(0);

  // Check service availability on mount
  useEffect(() => {
    checkStatus();
  }, []);

  // Auto-benchmark if molecule ID is provided
  useEffect(() => {
    if (selectedMoleculeId && view === 'single' && serviceAvailable) {
      benchmarkById(selectedMoleculeId);
    }
  }, [selectedMoleculeId, serviceAvailable]);

  const checkStatus = async () => {
    try {
      const status = await benchmarkApi.getStatus();
      setServiceAvailable(status.available);
      setReferenceCount(status.reference_compounds);
    } catch {
      setServiceAvailable(false);
    }
  };

  const benchmarkSmiles = async () => {
    if (!smilesInput.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const result = await benchmarkApi.benchmarkMolecule(smilesInput);
      setSingleResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Benchmark failed');
    } finally {
      setLoading(false);
    }
  };

  const benchmarkById = async (id: number) => {
    setLoading(true);
    setError(null);

    try {
      const result = await benchmarkApi.benchmarkMoleculeById(id);
      setSingleResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Benchmark failed');
    } finally {
      setLoading(false);
    }
  };

  const runBatchBenchmark = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await benchmarkApi.benchmarkBatch(undefined, topN, minScore);
      setBatchResults(response.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Batch benchmark failed');
    } finally {
      setLoading(false);
    }
  };

  const generateReport = async () => {
    setLoading(true);
    setError(null);

    try {
      const reportData = await benchmarkApi.generateReport();
      setReport(reportData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Report generation failed');
    } finally {
      setLoading(false);
    }
  };

  const loadReferences = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await benchmarkApi.getReferences();
      setReferences(data.compounds);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load references');
    } finally {
      setLoading(false);
    }
  };

  // Load references when switching to that view
  useEffect(() => {
    if (view === 'references' && references.length === 0 && serviceAvailable) {
      loadReferences();
    }
  }, [view, serviceAvailable]);

  if (serviceAvailable === null) {
    return (
      <div className="benchmark-panel bg-gray-800 rounded-lg p-6 text-center">
        <div className="loading-spinner mx-auto mb-4" />
        <p className="text-gray-400">Checking benchmark service...</p>
      </div>
    );
  }

  if (!serviceAvailable) {
    return (
      <div className="benchmark-panel bg-gray-800 rounded-lg p-6 text-center">
        <AlertCircle size={48} className="text-yellow-500 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-white mb-2">Benchmark Service Unavailable</h3>
        <p className="text-gray-400">The reference database has not been loaded.</p>
        <button
          onClick={checkStatus}
          className="mt-4 px-4 py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="benchmark-panel bg-gray-800 rounded-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <BarChart3 size={20} className="text-purple-400" />
          <h2 className="text-lg font-semibold text-white">Benchmark Analysis</h2>
          <span className="text-xs text-gray-500">({referenceCount} references)</span>
        </div>
      </div>

      {/* View Tabs */}
      <div className="flex gap-2 mb-6 border-b border-gray-700 pb-4">
        {[
          { id: 'single', label: 'Single Molecule' },
          { id: 'batch', label: 'Batch Analysis' },
          { id: 'report', label: 'Full Report' },
          { id: 'references', label: 'References' }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setView(tab.id as BenchmarkView)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              view === tab.id
                ? 'bg-purple-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-900/50 border border-red-500 rounded-lg text-red-300">
          {error}
        </div>
      )}

      {/* Single Molecule Benchmark */}
      {view === 'single' && (
        <div className="single-benchmark">
          <div className="flex gap-2 mb-4">
            <input
              type="text"
              value={smilesInput}
              onChange={(e) => setSmilesInput(e.target.value)}
              placeholder="Enter SMILES to benchmark"
              className="flex-1 px-4 py-2 rounded-lg bg-gray-700 text-white border-2 border-gray-600
                focus:outline-none focus:border-purple-500"
            />
            <button
              onClick={benchmarkSmiles}
              disabled={loading || !smilesInput.trim()}
              className={`px-6 py-2 rounded-lg font-semibold transition-colors ${
                loading || !smilesInput.trim()
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  : 'bg-purple-600 text-white hover:bg-purple-700'
              }`}
            >
              {loading ? 'Analyzing...' : 'Benchmark'}
            </button>
          </div>

          {selectedMoleculeId && (
            <p className="text-sm text-gray-400 mb-4">
              Or viewing benchmark for molecule ID: {selectedMoleculeId}
            </p>
          )}

          {singleResult && <BenchmarkResultView result={singleResult} />}
        </div>
      )}

      {/* Batch Benchmark */}
      {view === 'batch' && (
        <div className="batch-benchmark">
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Top N Results: {topN}
              </label>
              <input
                type="range"
                min="5"
                max="50"
                value={topN}
                onChange={(e) => setTopN(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Minimum Score: {minScore}
              </label>
              <input
                type="range"
                min="0"
                max="80"
                step="5"
                value={minScore}
                onChange={(e) => setMinScore(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
          </div>

          <button
            onClick={runBatchBenchmark}
            disabled={loading}
            className={`w-full px-6 py-3 rounded-lg font-semibold mb-4 transition-colors ${
              loading
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-purple-600 text-white hover:bg-purple-700'
            }`}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <div className="loading-spinner" style={{ width: 16, height: 16 }} />
                Analyzing...
              </span>
            ) : (
              'Run Batch Benchmark'
            )}
          </button>

          {batchResults.length > 0 && (
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {batchResults.map((result, index) => (
                <div
                  key={result.molecule_id || index}
                  onClick={() => result.molecule_id && onSelectMolecule?.(result.molecule_id)}
                  className="p-3 bg-gray-700 rounded-lg cursor-pointer hover:bg-gray-600 transition-colors"
                >
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-white">
                      #{result.molecule_id} - {result.scaffold_type.replace(/_/g, ' ')}
                    </span>
                    <span className={`px-2 py-1 rounded text-sm font-semibold ${
                      result.overall_score >= 70 ? 'bg-emerald-600' :
                      result.overall_score >= 50 ? 'bg-yellow-600' : 'bg-red-600'
                    } text-white`}>
                      {result.overall_score.toFixed(0)}
                    </span>
                  </div>
                  <p className="text-sm text-gray-400 mt-1">{result.recommendation}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Full Report */}
      {view === 'report' && (
        <div className="full-report">
          <button
            onClick={generateReport}
            disabled={loading}
            className={`w-full px-6 py-3 rounded-lg font-semibold mb-4 transition-colors ${
              loading
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-purple-600 text-white hover:bg-purple-700'
            }`}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <div className="loading-spinner" style={{ width: 16, height: 16 }} />
                Generating Report...
              </span>
            ) : (
              'Generate Benchmark Report'
            )}
          </button>

          {report && <BenchmarkReport report={report} onSelectMolecule={onSelectMolecule} />}
        </div>
      )}

      {/* Reference Compounds */}
      {view === 'references' && (
        <div className="reference-compounds">
          <h3 className="text-lg font-semibold text-white mb-4">
            Reference Quaternary Ammonium Compounds
          </h3>

          {loading ? (
            <div className="text-center py-8">
              <div className="loading-spinner mx-auto mb-2" />
              <span className="text-gray-400">Loading references...</span>
            </div>
          ) : references.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              No reference compounds available
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-4 max-h-96 overflow-y-auto">
              {references.map((ref, index) => (
                <ReferenceCompoundCard key={index} compound={ref} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default BenchmarkPanel;
