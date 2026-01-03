/**
 * ReferenceCompoundCard - Display reference compound information
 */

import React, { useState } from 'react';
import { ChevronDown, ChevronUp, FlaskConical, Activity, ExternalLink } from 'lucide-react';
import { ReferenceCompound } from '../../services/api';

interface ReferenceCompoundCardProps {
  compound: ReferenceCompound;
}

export const ReferenceCompoundCard: React.FC<ReferenceCompoundCardProps> = ({ compound }) => {
  const [expanded, setExpanded] = useState(false);

  const hasMicData = compound.mic_s_aureus || compound.mic_e_coli ||
                     compound.mic_p_aeruginosa || compound.mic_c_albicans;

  return (
    <div className="reference-compound-card bg-gray-700 rounded-lg overflow-hidden">
      {/* Header */}
      <div
        className="p-4 cursor-pointer hover:bg-gray-600 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-purple-600 rounded-lg flex items-center justify-center">
              <FlaskConical size={20} className="text-white" />
            </div>
            <div>
              <h4 className="font-semibold text-white">{compound.name}</h4>
              {compound.chembl_id && (
                <a
                  href={`https://www.ebi.ac.uk/chembl/compound_report_card/${compound.chembl_id}/`}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  className="text-sm text-blue-400 hover:text-blue-300 flex items-center gap-1"
                >
                  {compound.chembl_id}
                  <ExternalLink size={12} />
                </a>
              )}
            </div>
          </div>
          <div className="flex items-center gap-3">
            {compound.ld50_oral_rat && (
              <div className="text-right">
                <div className="text-xs text-gray-400">LD50</div>
                <div className="text-sm text-white">{compound.ld50_oral_rat} mg/kg</div>
              </div>
            )}
            {expanded ? (
              <ChevronUp size={20} className="text-gray-400" />
            ) : (
              <ChevronDown size={20} className="text-gray-400" />
            )}
          </div>
        </div>
      </div>

      {/* Expanded Content */}
      {expanded && (
        <div className="px-4 pb-4 border-t border-gray-600 pt-4">
          {/* SMILES */}
          <div className="mb-4">
            <label className="text-xs text-gray-400 block mb-1">SMILES</label>
            <code className="text-xs text-gray-300 font-mono break-all block p-2 bg-gray-800 rounded">
              {compound.smiles}
            </code>
          </div>

          {/* MIC Data */}
          {hasMicData && (
            <div className="mb-4">
              <div className="flex items-center gap-2 mb-2">
                <Activity size={14} className="text-purple-400" />
                <span className="text-sm font-medium text-gray-300">
                  Antimicrobial Activity (MIC µg/mL)
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {compound.mic_s_aureus && (
                  <div className="p-2 bg-gray-800 rounded">
                    <div className="text-xs text-gray-400">S. aureus</div>
                    <div className="text-sm text-white font-medium">
                      {compound.mic_s_aureus} µg/mL
                    </div>
                  </div>
                )}
                {compound.mic_e_coli && (
                  <div className="p-2 bg-gray-800 rounded">
                    <div className="text-xs text-gray-400">E. coli</div>
                    <div className="text-sm text-white font-medium">
                      {compound.mic_e_coli} µg/mL
                    </div>
                  </div>
                )}
                {compound.mic_p_aeruginosa && (
                  <div className="p-2 bg-gray-800 rounded">
                    <div className="text-xs text-gray-400">P. aeruginosa</div>
                    <div className="text-sm text-white font-medium">
                      {compound.mic_p_aeruginosa} µg/mL
                    </div>
                  </div>
                )}
                {compound.mic_c_albicans && (
                  <div className="p-2 bg-gray-800 rounded">
                    <div className="text-xs text-gray-400">C. albicans</div>
                    <div className="text-sm text-white font-medium">
                      {compound.mic_c_albicans} µg/mL
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Applications */}
          {compound.applications && compound.applications.length > 0 && (
            <div>
              <span className="text-xs text-gray-400 block mb-2">Applications</span>
              <div className="flex flex-wrap gap-2">
                {compound.applications.map((app, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 bg-purple-900/50 text-purple-300 rounded text-xs"
                  >
                    {app}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ReferenceCompoundCard;
