/**
 * SearchResultCard - Card displaying a single search result
 */

import React from 'react';
import { SearchResult, SimilarityResult } from '../../services/api';
import { MoleculeImage } from '../molecules/MoleculeImage';

interface SearchResultCardProps {
  result: SearchResult | SimilarityResult;
  searchMode: 'substructure' | 'similarity';
  onSelect?: () => void;
}

// Type guard to check if result is SimilarityResult
function isSimilarityResult(result: SearchResult | SimilarityResult): result is SimilarityResult {
  return 'similarity' in result;
}

export const SearchResultCard: React.FC<SearchResultCardProps> = ({
  result,
  searchMode,
  onSelect
}) => {
  const isSimilarity = isSimilarityResult(result);

  const scoreDisplay = isSimilarity
    ? `${(result.similarity * 100).toFixed(0)}% similar`
    : `${(result as SearchResult).match_count} match${(result as SearchResult).match_count !== 1 ? 'es' : ''}`;

  const scoreColor = isSimilarity
    ? result.similarity >= 0.8 ? 'bg-emerald-600' :
      result.similarity >= 0.7 ? 'bg-yellow-600' : 'bg-gray-600'
    : 'bg-blue-600';

  // Get scores from result
  const efficacy = isSimilarity
    ? result.scores.efficacy_score
    : result.scores.efficacy;
  const safety = isSimilarity
    ? result.scores.safety_score
    : result.scores.safety;

  return (
    <div
      className="search-result-card bg-gray-700 rounded-lg p-3 hover:bg-gray-600
        transition-all cursor-pointer"
      onClick={onSelect}
    >
      {/* Structure Image */}
      <div className="mb-2 bg-gray-900 rounded overflow-hidden">
        <MoleculeImage
          smiles={result.smiles}
          width={200}
          height={120}
          highlightQuat={true}
          className="w-full"
        />
      </div>

      {/* Match Score */}
      <div className="flex justify-between items-center mb-2">
        <span className={`text-xs font-medium px-2 py-1 rounded ${scoreColor} text-white`}>
          {scoreDisplay}
        </span>
        {result.molecule_id && (
          <span className="text-xs text-gray-400">
            ID: {result.molecule_id}
          </span>
        )}
      </div>

      {/* Properties */}
      <div className="grid grid-cols-2 gap-1 text-xs mb-2">
        {efficacy !== undefined && (
          <div className="text-gray-300">
            Eff: <span className={`font-medium ${
              efficacy >= 70 ? 'text-emerald-400' :
              efficacy >= 50 ? 'text-yellow-400' : 'text-red-400'
            }`}>
              {efficacy?.toFixed(0)}
            </span>
          </div>
        )}
        {safety !== undefined && (
          <div className="text-gray-300">
            Safe: <span className={`font-medium ${
              safety >= 70 ? 'text-emerald-400' :
              safety >= 50 ? 'text-yellow-400' : 'text-red-400'
            }`}>
              {safety?.toFixed(0)}
            </span>
          </div>
        )}
      </div>

      {/* SMILES (truncated) */}
      <div className="text-xs text-gray-500 font-mono truncate" title={result.smiles}>
        {result.smiles}
      </div>

      {/* Matched atoms indicator for substructure search */}
      {searchMode === 'substructure' && !isSimilarity && (result as SearchResult).match_atoms?.length > 0 && (
        <div className="mt-1 text-xs text-blue-400">
          {(result as SearchResult).match_atoms.reduce((acc, arr) => acc + arr.length, 0)} atoms matched
        </div>
      )}

      {/* Name if available */}
      {result.name && (
        <div className="mt-1 text-xs text-gray-400 truncate" title={result.name}>
          {result.name}
        </div>
      )}
    </div>
  );
};

export default SearchResultCard;
