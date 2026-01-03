/**
 * PatternSelector - Grid of common SMARTS pattern templates
 */

import React, { useState } from 'react';
import { X } from 'lucide-react';
import { PatternsResponse } from '../../services/api';

interface PatternSelectorProps {
  patterns: PatternsResponse;
  onSelect: (smarts: string) => void;
  onClose: () => void;
}

// Pattern descriptions for common patterns
const patternDescriptions: Record<string, string> = {
  QUAT_ALIPHATIC: 'Aliphatic quaternary nitrogen (sp3, 4 bonds)',
  QUAT_AROMATIC: 'Aromatic quaternary nitrogen (pyridinium)',
  QUAT_ANY: 'Any quaternary nitrogen (not nitro)',
  LONG_ALKYL_CHAIN: 'Long alkyl chain (8+ carbons)',
  MEDIUM_ALKYL_CHAIN: 'Medium alkyl chain (6+ carbons)',
  BENZYL: 'Benzyl group (CH2 attached to phenyl)',
  PHENYL: 'Phenyl ring (benzene)',
  HYDROXYL: 'Hydroxyl group (OH)',
  ETHER: 'Ether linkage (C-O-C)',
  ESTER: 'Ester group',
  AMIDE: 'Amide group',
  BAC_CORE: 'Benzalkonium core structure',
  DDAC_CORE: 'Didecyl quaternary core',
  CETYL_CORE: 'Cetyl quaternary core',
  CHLORO: 'Chlorine atom',
  BROMO: 'Bromine atom',
  FLUORO: 'Fluorine atom',
  IODO: 'Iodine atom',
  PYRIDINIUM: 'Pyridinium ring',
  IMIDAZOLIUM: 'Imidazolium ring',
  MORPHOLINE: 'Morpholine ring',
  aliphatic_quat: 'Aliphatic quaternary nitrogen',
  aromatic_quat: 'Aromatic quaternary nitrogen',
  any_quat: 'Any type of quaternary nitrogen',
  bac_core: 'Benzalkonium-like core',
  ddac_core: 'Didecyl quaternary core',
  cetyl_core: 'Cetyl quaternary core'
};

export const PatternSelector: React.FC<PatternSelectorProps> = ({
  patterns,
  onSelect,
  onClose
}) => {
  const [selectedCategory, setSelectedCategory] = useState<'all' | 'quat'>('quat');

  const currentPatterns = selectedCategory === 'quat'
    ? patterns.quat_patterns
    : patterns.all_patterns;

  return (
    <div className="pattern-selector bg-gray-700 rounded-lg p-4 border border-gray-600">
      {/* Header */}
      <div className="flex justify-between items-center mb-3">
        <h4 className="font-semibold text-white">Common Patterns</h4>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-white transition-colors"
        >
          <X size={18} />
        </button>
      </div>

      {/* Category Filter */}
      <div className="flex gap-2 mb-3">
        <button
          onClick={() => setSelectedCategory('quat')}
          className={`px-3 py-1 text-sm rounded-full transition-colors
            ${selectedCategory === 'quat'
              ? 'bg-emerald-600 text-white'
              : 'bg-gray-600 text-gray-300 hover:bg-gray-500'}`}
        >
          Quaternary Ammonium
        </button>
        <button
          onClick={() => setSelectedCategory('all')}
          className={`px-3 py-1 text-sm rounded-full transition-colors
            ${selectedCategory === 'all'
              ? 'bg-emerald-600 text-white'
              : 'bg-gray-600 text-gray-300 hover:bg-gray-500'}`}
        >
          All Patterns
        </button>
      </div>

      {/* Pattern Grid */}
      <div className="grid grid-cols-2 gap-2 max-h-64 overflow-y-auto">
        {Object.entries(currentPatterns).map(([name, smarts]) => (
          <button
            key={name}
            onClick={() => onSelect(smarts)}
            className="text-left p-2 bg-gray-600 rounded hover:bg-gray-500 transition-colors"
          >
            <div className="font-medium text-white text-sm">
              {name.replace(/_/g, ' ')}
            </div>
            <div className="text-xs text-emerald-400 font-mono truncate" title={smarts}>
              {smarts}
            </div>
            {patternDescriptions[name] && (
              <div className="text-xs text-gray-400 mt-1 line-clamp-2">
                {patternDescriptions[name]}
              </div>
            )}
          </button>
        ))}
      </div>

      {Object.keys(currentPatterns).length === 0 && (
        <div className="text-center py-4 text-gray-400">
          No patterns available
        </div>
      )}

      {/* Quick Reference */}
      <div className="mt-3 pt-3 border-t border-gray-600">
        <p className="text-xs text-gray-400">
          <strong className="text-gray-300">Tip:</strong> Click a pattern to use it.
          SMARTS is similar to SMILES but with wildcards.
        </p>
      </div>
    </div>
  );
};

export default PatternSelector;
