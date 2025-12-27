import React, { useEffect } from 'react';
import { GeneratorControls } from './components/GeneratorControls';
import { ResultsTable } from './components/ResultsTable';
import { MoleculeDetail } from './components/MoleculeDetail';
import { StatusBar } from './components/StatusBar';
import { ExportModal } from './components/ExportModal';
import { useStore } from './hooks/useStore';
import { useBackend } from './hooks/useBackend';

function App() {
  const { 
    selectedMolecule, 
    showExportModal, 
    setShowExportModal 
  } = useStore();
  
  const { checkConnection } = useBackend();

  useEffect(() => {
    checkConnection();
  }, []);

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-gray-100">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">Q</span>
            </div>
            <h1 className="text-xl font-semibold">Quat Generator Pro</h1>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowExportModal(true)}
              className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm"
            >
              Export
            </button>
          </div>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel - Controls */}
        <div className="w-80 border-r border-gray-700 overflow-y-auto">
          <GeneratorControls />
        </div>

        {/* Center - Results table */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <ResultsTable />
        </div>

        {/* Right panel - Detail view */}
        {selectedMolecule && (
          <div className="w-96 border-l border-gray-700 overflow-y-auto">
            <MoleculeDetail molecule={selectedMolecule} />
          </div>
        )}
      </div>

      {/* Status bar */}
      <StatusBar />

      {/* Modals */}
      {showExportModal && (
        <ExportModal onClose={() => setShowExportModal(false)} />
      )}
    </div>
  );
}

export default App;
