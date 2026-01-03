/**
 * BenchmarkReport - Full benchmark report summary display
 */

import React from 'react';
import { FileText, TrendingUp, Users, Target, Lightbulb, PieChart } from 'lucide-react';
import { BenchmarkReportSummary } from '../../services/api';

interface BenchmarkReportProps {
  report: BenchmarkReportSummary;
  onSelectMolecule?: (id: number) => void;
}

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

export const BenchmarkReport: React.FC<BenchmarkReportProps> = ({ report, onSelectMolecule }) => {
  const { summary, scaffold_distribution, top_candidates, reference_comparison, recommendations } = report;

  const totalComparisons = reference_comparison.total_comparisons || 1;
  const betterPercent = (reference_comparison.better_than_reference / totalComparisons) * 100;
  const similarPercent = (reference_comparison.similar_to_reference / totalComparisons) * 100;
  const worsePercent = (reference_comparison.worse_than_reference / totalComparisons) * 100;

  return (
    <div className="benchmark-report space-y-6">
      {/* Report Header */}
      <div className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
        <div className="flex items-center gap-3">
          <FileText size={24} className="text-purple-400" />
          <div>
            <h3 className="text-lg font-semibold text-white">Benchmark Report</h3>
            <p className="text-sm text-gray-400">
              Generated: {new Date(report.generated_at).toLocaleString()}
            </p>
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="p-4 bg-gray-700 rounded-lg text-center">
          <div className="text-2xl font-bold text-white">{summary.total_molecules}</div>
          <div className="text-sm text-gray-400">Total Molecules</div>
        </div>
        <div className="p-4 bg-gray-700 rounded-lg text-center">
          <div className="text-2xl font-bold text-blue-400">{summary.molecules_benchmarked}</div>
          <div className="text-sm text-gray-400">Benchmarked</div>
        </div>
        <div className="p-4 bg-gray-700 rounded-lg text-center">
          <div className={`text-2xl font-bold ${getScoreColor(summary.avg_overall_score)}`}>
            {summary.avg_overall_score.toFixed(1)}
          </div>
          <div className="text-sm text-gray-400">Avg Score</div>
        </div>
        <div className="p-4 bg-gray-700 rounded-lg text-center">
          <div className="text-2xl font-bold text-emerald-400">{summary.top_candidates_count}</div>
          <div className="text-sm text-gray-400">Top Candidates</div>
        </div>
      </div>

      {/* Reference Comparison Overview */}
      <div className="p-4 bg-gray-700 rounded-lg">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp size={18} className="text-purple-400" />
          <h4 className="font-semibold text-white">Reference Comparison Overview</h4>
        </div>

        <div className="space-y-3">
          {/* Progress bars */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-emerald-400">Better than reference</span>
              <span className="text-gray-400">{reference_comparison.better_than_reference} ({betterPercent.toFixed(1)}%)</span>
            </div>
            <div className="h-2 bg-gray-600 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                style={{ width: `${betterPercent}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-yellow-400">Similar to reference</span>
              <span className="text-gray-400">{reference_comparison.similar_to_reference} ({similarPercent.toFixed(1)}%)</span>
            </div>
            <div className="h-2 bg-gray-600 rounded-full overflow-hidden">
              <div
                className="h-full bg-yellow-500 rounded-full transition-all duration-500"
                style={{ width: `${similarPercent}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-red-400">Worse than reference</span>
              <span className="text-gray-400">{reference_comparison.worse_than_reference} ({worsePercent.toFixed(1)}%)</span>
            </div>
            <div className="h-2 bg-gray-600 rounded-full overflow-hidden">
              <div
                className="h-full bg-red-500 rounded-full transition-all duration-500"
                style={{ width: `${worsePercent}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Scaffold Distribution */}
      {Object.keys(scaffold_distribution).length > 0 && (
        <div className="p-4 bg-gray-700 rounded-lg">
          <div className="flex items-center gap-2 mb-4">
            <PieChart size={18} className="text-purple-400" />
            <h4 className="font-semibold text-white">Scaffold Distribution</h4>
          </div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(scaffold_distribution).map(([scaffold, count]) => (
              <div
                key={scaffold}
                className="px-3 py-2 bg-gray-600 rounded-lg flex items-center gap-2"
              >
                <span className="text-white">{scaffold.replace(/_/g, ' ')}</span>
                <span className="px-2 py-0.5 bg-gray-500 rounded text-sm text-gray-300">
                  {count}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Property Improvements/Deficits */}
      <div className="grid grid-cols-2 gap-4">
        {Object.keys(reference_comparison.property_improvements || {}).length > 0 && (
          <div className="p-4 bg-gray-700 rounded-lg">
            <h4 className="font-semibold text-emerald-400 mb-3">Property Improvements</h4>
            <div className="space-y-2">
              {Object.entries(reference_comparison.property_improvements).map(([prop, count]) => (
                <div key={prop} className="flex justify-between text-sm">
                  <span className="text-gray-300">{prop}</span>
                  <span className="text-emerald-400">+{count}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {Object.keys(reference_comparison.property_deficits || {}).length > 0 && (
          <div className="p-4 bg-gray-700 rounded-lg">
            <h4 className="font-semibold text-red-400 mb-3">Property Deficits</h4>
            <div className="space-y-2">
              {Object.entries(reference_comparison.property_deficits).map(([prop, count]) => (
                <div key={prop} className="flex justify-between text-sm">
                  <span className="text-gray-300">{prop}</span>
                  <span className="text-red-400">-{count}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Top Candidates */}
      {top_candidates.length > 0 && (
        <div className="p-4 bg-gray-700 rounded-lg">
          <div className="flex items-center gap-2 mb-4">
            <Target size={18} className="text-purple-400" />
            <h4 className="font-semibold text-white">Top Candidates</h4>
          </div>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {top_candidates.map((candidate, index) => (
              <div
                key={candidate.molecule_id || index}
                onClick={() => candidate.molecule_id && onSelectMolecule?.(candidate.molecule_id)}
                className="p-3 bg-gray-600 rounded-lg cursor-pointer hover:bg-gray-500 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-white font-medium">
                      #{candidate.molecule_id}
                    </span>
                    <span className="text-gray-400 text-sm ml-2">
                      {candidate.scaffold_type?.replace(/_/g, ' ')}
                    </span>
                  </div>
                  <span className={`px-2 py-1 rounded text-sm font-semibold ${getScoreBgColor(candidate.overall_score)} text-white`}>
                    {candidate.overall_score.toFixed(0)}
                  </span>
                </div>
                {candidate.recommendation && (
                  <p className="text-sm text-gray-400 mt-1 truncate">{candidate.recommendation}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Closest References Used */}
      {Object.keys(reference_comparison.closest_references_used || {}).length > 0 && (
        <div className="p-4 bg-gray-700 rounded-lg">
          <div className="flex items-center gap-2 mb-4">
            <Users size={18} className="text-purple-400" />
            <h4 className="font-semibold text-white">Reference Compounds Used</h4>
          </div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(reference_comparison.closest_references_used)
              .sort(([, a], [, b]) => b - a)
              .map(([ref, count]) => (
                <div
                  key={ref}
                  className="px-3 py-1 bg-gray-600 rounded-full flex items-center gap-2 text-sm"
                >
                  <span className="text-white">{ref}</span>
                  <span className="text-gray-400">×{count}</span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="p-4 bg-purple-900/30 rounded-lg border border-purple-500/30">
          <div className="flex items-center gap-2 mb-3">
            <Lightbulb size={18} className="text-purple-400" />
            <h4 className="font-semibold text-white">Recommendations</h4>
          </div>
          <ul className="space-y-2">
            {recommendations.map((rec, index) => (
              <li key={index} className="text-gray-300 text-sm flex items-start gap-2">
                <span className="text-purple-400 mt-1">•</span>
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default BenchmarkReport;
