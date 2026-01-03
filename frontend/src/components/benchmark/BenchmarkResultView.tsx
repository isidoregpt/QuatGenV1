/**
 * BenchmarkResultView - Detailed view of a single benchmark result
 */

import React from 'react';
import { CheckCircle, XCircle, MinusCircle, TrendingUp, TrendingDown, Award, AlertTriangle } from 'lucide-react';
import { BenchmarkResult, PropertyComparison } from '../../services/api';

interface BenchmarkResultViewProps {
  result: BenchmarkResult;
  compact?: boolean;
}

const OutcomeIcon: React.FC<{ outcome: PropertyComparison['outcome'] }> = ({ outcome }) => {
  switch (outcome) {
    case 'better':
      return <TrendingUp size={16} className="text-emerald-400" />;
    case 'worse':
      return <TrendingDown size={16} className="text-red-400" />;
    case 'similar':
      return <MinusCircle size={16} className="text-yellow-400" />;
    default:
      return <MinusCircle size={16} className="text-gray-400" />;
  }
};

const getScoreColor = (score: number): string => {
  if (score >= 70) return 'text-emerald-400';
  if (score >= 50) return 'text-yellow-400';
  return 'text-red-400';
};

const getScoreBgColor = (score: number): string => {
  if (score >= 70) return 'bg-emerald-600';
  if (score >= 50) return 'bg-yellow-600';
  return 'bg-red-600';
};

export const BenchmarkResultView: React.FC<BenchmarkResultViewProps> = ({ result, compact = false }) => {
  if (compact) {
    return (
      <div className="benchmark-result-compact p-4 bg-gray-700 rounded-lg">
        <div className="flex items-center justify-between mb-2">
          <span className="text-white font-medium">{result.scaffold_type.replace(/_/g, ' ')}</span>
          <span className={`text-2xl font-bold ${getScoreColor(result.overall_score)}`}>
            {result.overall_score.toFixed(0)}
          </span>
        </div>
        <p className="text-sm text-gray-400">{result.recommendation}</p>
      </div>
    );
  }

  return (
    <div className="benchmark-result-view space-y-6">
      {/* Header with Overall Score */}
      <div className="flex items-start justify-between p-4 bg-gray-700 rounded-lg">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <Award size={20} className="text-purple-400" />
            <h3 className="text-lg font-semibold text-white">Benchmark Results</h3>
          </div>
          <p className="text-sm text-gray-400 font-mono break-all">{result.smiles}</p>
          <div className="mt-2 flex items-center gap-4">
            <span className="text-sm text-gray-400">
              Scaffold: <span className="text-white">{result.scaffold_type.replace(/_/g, ' ')}</span>
            </span>
            <span className="text-sm text-gray-400">
              Confidence: <span className="text-white">{(result.confidence * 100).toFixed(0)}%</span>
            </span>
            <span className="text-sm text-gray-400">
              Novelty: <span className="text-white">{(result.structural_novelty * 100).toFixed(0)}%</span>
            </span>
          </div>
        </div>
        <div className="text-center ml-4">
          <div className={`text-4xl font-bold ${getScoreColor(result.overall_score)}`}>
            {result.overall_score.toFixed(0)}
          </div>
          <div className="text-xs text-gray-400 mt-1">Overall Score</div>
        </div>
      </div>

      {/* Recommendation */}
      <div className="p-4 bg-gray-700/50 rounded-lg border-l-4 border-purple-500">
        <p className="text-white">{result.recommendation}</p>
      </div>

      {/* Property Comparison Summary */}
      <div className="grid grid-cols-3 gap-4">
        <div className="p-4 bg-emerald-900/30 rounded-lg text-center">
          <div className="text-2xl font-bold text-emerald-400">{result.properties_better}</div>
          <div className="text-sm text-gray-400">Better</div>
        </div>
        <div className="p-4 bg-yellow-900/30 rounded-lg text-center">
          <div className="text-2xl font-bold text-yellow-400">{result.properties_similar}</div>
          <div className="text-sm text-gray-400">Similar</div>
        </div>
        <div className="p-4 bg-red-900/30 rounded-lg text-center">
          <div className="text-2xl font-bold text-red-400">{result.properties_worse}</div>
          <div className="text-sm text-gray-400">Worse</div>
        </div>
      </div>

      {/* Closest Reference Compounds */}
      {result.closest_references.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
            Closest Reference Compounds
          </h4>
          <div className="grid gap-3">
            {result.closest_references.map((ref, index) => (
              <div key={index} className="p-3 bg-gray-700 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-white">{ref.name}</span>
                  <span className="px-2 py-1 bg-gray-600 rounded text-sm text-gray-300">
                    {(ref.similarity * 100).toFixed(0)}% similar
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-sm">
                  {ref.mic_s_aureus && (
                    <div>
                      <span className="text-gray-400">MIC S.aureus:</span>
                      <span className="text-white ml-1">{ref.mic_s_aureus} µg/mL</span>
                    </div>
                  )}
                  {ref.mic_e_coli && (
                    <div>
                      <span className="text-gray-400">MIC E.coli:</span>
                      <span className="text-white ml-1">{ref.mic_e_coli} µg/mL</span>
                    </div>
                  )}
                  {ref.ld50 && (
                    <div>
                      <span className="text-gray-400">LD50:</span>
                      <span className="text-white ml-1">{ref.ld50} mg/kg</span>
                    </div>
                  )}
                </div>
                {ref.applications && ref.applications.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {ref.applications.map((app, i) => (
                      <span key={i} className="px-2 py-0.5 bg-gray-600 rounded text-xs text-gray-300">
                        {app}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Property Comparisons Table */}
      {result.property_comparisons.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
            Property Comparisons
          </h4>
          <div className="overflow-hidden rounded-lg border border-gray-600">
            <table className="w-full text-sm">
              <thead className="bg-gray-700">
                <tr>
                  <th className="px-4 py-2 text-left text-gray-300">Property</th>
                  <th className="px-4 py-2 text-right text-gray-300">Generated</th>
                  <th className="px-4 py-2 text-right text-gray-300">Reference</th>
                  <th className="px-4 py-2 text-center text-gray-300">Outcome</th>
                </tr>
              </thead>
              <tbody>
                {result.property_comparisons.map((comp, index) => (
                  <tr key={index} className="border-t border-gray-600">
                    <td className="px-4 py-2 text-white">{comp.property}</td>
                    <td className="px-4 py-2 text-right text-white font-mono">
                      {comp.generated.toFixed(2)}
                    </td>
                    <td className="px-4 py-2 text-right text-gray-400 font-mono">
                      {comp.reference.toFixed(2)}
                    </td>
                    <td className="px-4 py-2">
                      <div className="flex items-center justify-center gap-2">
                        <OutcomeIcon outcome={comp.outcome} />
                        <span className={`text-xs ${
                          comp.outcome === 'better' ? 'text-emerald-400' :
                          comp.outcome === 'worse' ? 'text-red-400' :
                          comp.outcome === 'similar' ? 'text-yellow-400' :
                          'text-gray-400'
                        }`}>
                          {comp.outcome}
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Advantages and Disadvantages */}
      <div className="grid grid-cols-2 gap-4">
        {result.advantages.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-semibold text-emerald-400 flex items-center gap-2">
              <CheckCircle size={16} />
              Advantages
            </h4>
            <ul className="space-y-1">
              {result.advantages.map((adv, index) => (
                <li key={index} className="text-sm text-gray-300 flex items-start gap-2">
                  <span className="text-emerald-400 mt-1">•</span>
                  {adv}
                </li>
              ))}
            </ul>
          </div>
        )}
        {result.disadvantages.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-semibold text-red-400 flex items-center gap-2">
              <AlertTriangle size={16} />
              Disadvantages
            </h4>
            <ul className="space-y-1">
              {result.disadvantages.map((dis, index) => (
                <li key={index} className="text-sm text-gray-300 flex items-start gap-2">
                  <span className="text-red-400 mt-1">•</span>
                  {dis}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default BenchmarkResultView;
