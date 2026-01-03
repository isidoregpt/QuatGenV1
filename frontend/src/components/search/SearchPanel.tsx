/**
 * SearchPanel - Main search interface for molecules
 */

import React, { useState, useEffect } from 'react';
import { Search, X } from 'lucide-react';
import { searchApi, SearchResult, SimilarityResult, PatternsResponse } from '../../services/api';
import { PatternSelector } from './PatternSelector';
import { SearchResultCard } from './SearchResultCard';

type SearchMode = 'substructure' | 'similarity';

interface SearchPanelProps {
  onSelectMolecule?: (moleculeId: number) => void;
  onClose?: () => void;
}

export const SearchPanel: React.FC<SearchPanelProps> = ({ onSelectMolecule, onClose }) => {
  // Search mode
  const [mode, setMode] = useState<SearchMode>('substructure');

  // Substructure search state
  const [smartsQuery, setSmartsQuery] = useState('');
  const [smartsValid, setSmartsValid] = useState<boolean | null>(null);
  const [smartsError, setSmartsError] = useState<string | null>(null);

  // Similarity search state
  const [smilesQuery, setSmilesQuery] = useState('');
  const [similarityThreshold, setSimilarityThreshold] = useState(0.7);

  // Common state
  const [maxResults, setMaxResults] = useState(50);
  const [searching, setSearching] = useState(false);
  const [substructureResults, setSubstructureResults] = useState<SearchResult[] | null>(null);
  const [similarityResults, setSimilarityResults] = useState<SimilarityResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Pattern templates
  const [patterns, setPatterns] = useState<PatternsResponse | null>(null);
  const [showPatterns, setShowPatterns] = useState(false);

  // Load patterns on mount
  useEffect(() => {
    loadPatterns();
  }, []);

  const loadPatterns = async () => {
    try {
      const data = await searchApi.getPatterns();
      setPatterns(data);
    } catch (err) {
      console.error('Failed to load patterns:', err);
    }
  };

  // Validate SMARTS as user types
  useEffect(() => {
    if (!smartsQuery.trim()) {
      setSmartsValid(null);
      setSmartsError(null);
      return;
    }

    const timer = setTimeout(async () => {
      try {
        const result = await searchApi.validateSmarts(smartsQuery);
        setSmartsValid(result.is_valid);
        setSmartsError(result.is_valid ? null : result.error);
      } catch {
        setSmartsValid(false);
        setSmartsError('Validation failed');
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [smartsQuery]);

  const handleSearch = async () => {
    setSearching(true);
    setError(null);
    setSubstructureResults(null);
    setSimilarityResults(null);

    try {
      if (mode === 'substructure') {
        if (!smartsQuery.trim() || !smartsValid) {
          throw new Error('Invalid SMARTS pattern');
        }
        const results = await searchApi.substructureSearch({
          smarts: smartsQuery,
          max_results: maxResults
        });
        setSubstructureResults(results);
      } else {
        if (!smilesQuery.trim()) {
          throw new Error('Please enter a SMILES query');
        }
        const results = await searchApi.similaritySearch({
          smiles: smilesQuery,
          threshold: similarityThreshold,
          max_results: maxResults
        });
        setSimilarityResults(results);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setSearching(false);
    }
  };

  const handleSelectPattern = (smarts: string) => {
    setSmartsQuery(smarts);
    setShowPatterns(false);
  };

  const handleClear = () => {
    setSmartsQuery('');
    setSmilesQuery('');
    setSubstructureResults(null);
    setSimilarityResults(null);
    setError(null);
  };

  const resultsCount = mode === 'substructure'
    ? substructureResults?.length ?? 0
    : similarityResults?.length ?? 0;

  return (
    <div className="search-panel bg-gray-800 rounded-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <Search size={20} className="text-emerald-400" />
          <h2 className="text-lg font-semibold text-white">Molecule Search</h2>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X size={20} />
          </button>
        )}
      </div>

      {/* Mode Toggle */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setMode('substructure')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            mode === 'substructure'
              ? 'bg-emerald-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          Substructure
        </button>
        <button
          onClick={() => setMode('similarity')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            mode === 'similarity'
              ? 'bg-emerald-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          Similarity
        </button>
      </div>

      {/* Search Input */}
      {mode === 'substructure' ? (
        <div className="substructure-search mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            SMARTS Pattern
          </label>
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <input
                type="text"
                value={smartsQuery}
                onChange={(e) => setSmartsQuery(e.target.value)}
                placeholder="Enter SMARTS pattern (e.g., [N+])"
                className={`w-full px-4 py-2 rounded-lg bg-gray-700 text-white border-2
                  ${smartsValid === true ? 'border-emerald-500' :
                    smartsValid === false ? 'border-red-500' : 'border-gray-600'}
                  focus:outline-none focus:border-emerald-400`}
              />
              {smartsValid !== null && (
                <span className={`absolute right-3 top-1/2 -translate-y-1/2
                  ${smartsValid ? 'text-emerald-500' : 'text-red-500'}`}>
                  {smartsValid ? '✓' : '✗'}
                </span>
              )}
            </div>
            <button
              onClick={() => setShowPatterns(!showPatterns)}
              className="px-4 py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors"
              title="Show pattern templates"
            >
              📋
            </button>
          </div>
          {smartsError && (
            <p className="mt-1 text-sm text-red-400">{smartsError}</p>
          )}

          {/* Pattern Selector */}
          {showPatterns && patterns && (
            <div className="mt-2">
              <PatternSelector
                patterns={patterns}
                onSelect={handleSelectPattern}
                onClose={() => setShowPatterns(false)}
              />
            </div>
          )}
        </div>
      ) : (
        <div className="similarity-search mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Query SMILES
          </label>
          <input
            type="text"
            value={smilesQuery}
            onChange={(e) => setSmilesQuery(e.target.value)}
            placeholder="Enter SMILES of reference molecule"
            className="w-full px-4 py-2 rounded-lg bg-gray-700 text-white border-2 border-gray-600
              focus:outline-none focus:border-emerald-400"
          />

          <div className="mt-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Similarity Threshold: {(similarityThreshold * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.5"
              max="1.0"
              step="0.05"
              value={similarityThreshold}
              onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-400">
              <span>50% (More results)</span>
              <span>100% (Exact match)</span>
            </div>
          </div>
        </div>
      )}

      {/* Max Results */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Max Results: {maxResults}
        </label>
        <input
          type="range"
          min="10"
          max="100"
          step="10"
          value={maxResults}
          onChange={(e) => setMaxResults(parseInt(e.target.value))}
          className="w-full"
        />
      </div>

      {/* Search Button */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={handleSearch}
          disabled={searching || (mode === 'substructure' && !smartsValid)}
          className={`flex-1 px-6 py-3 rounded-lg font-semibold transition-colors
            ${searching || (mode === 'substructure' && !smartsValid)
              ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
              : 'bg-emerald-600 text-white hover:bg-emerald-700'}`}
        >
          {searching ? (
            <span className="flex items-center justify-center gap-2">
              <div className="loading-spinner" style={{ width: 16, height: 16 }} />
              Searching...
            </span>
          ) : (
            'Search'
          )}
        </button>
        <button
          onClick={handleClear}
          className="px-4 py-3 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors"
        >
          Clear
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-4 p-3 bg-red-900/50 border border-red-500 rounded-lg text-red-300">
          {error}
        </div>
      )}

      {/* Results */}
      {(substructureResults || similarityResults) && (
        <div className="search-results border-t border-gray-700 pt-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-white">
              Results ({resultsCount})
            </h3>
            <span className="text-sm text-gray-400">
              {mode === 'substructure' ? 'Substructure' : 'Similarity'} search
            </span>
          </div>

          {resultsCount === 0 ? (
            <div className="text-center py-8 text-gray-400">
              No matches found for your query
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4 max-h-96 overflow-y-auto pr-2">
              {mode === 'substructure' && substructureResults?.map((result, index) => (
                <SearchResultCard
                  key={result.molecule_id || index}
                  result={result}
                  searchMode="substructure"
                  onSelect={() => result.molecule_id && onSelectMolecule?.(result.molecule_id)}
                />
              ))}
              {mode === 'similarity' && similarityResults?.map((result, index) => (
                <SearchResultCard
                  key={result.molecule_id || index}
                  result={result}
                  searchMode="similarity"
                  onSelect={() => result.molecule_id && onSelectMolecule?.(result.molecule_id)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SearchPanel;
