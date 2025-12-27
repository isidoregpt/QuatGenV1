import React, { useEffect } from 'react';
import { GeneratorControls } from './components/GeneratorControls';
import { ResultsTable } from './components/ResultsTable';
import { MoleculeDetail } from './components/MoleculeDetail';
import { StatusBar } from './components/StatusBar';
import { useStore } from './hooks/useStore';
import { useBackend } from './hooks/useBackend';

function App() {
  const { selectedMolecule } = useStore();
  const { checkConnection } = useBackend();

  useEffect(() => { checkConnection(); }, []);

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-gray-100">
      <header className="bg-gray-800 border-b border-gray-700 px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">Q</span>
          </div>
          <h1 className="text-xl font-semibold">Quat Generator Pro</h1>
        </div>
      </header>
      <div className="flex-1 flex overflow-hidden">
        <div className="w-80 border-r border-gray-700 overflow-y-auto">
          <GeneratorControls />
        </div>
        <div className="flex-1 flex flex-col overflow-hidden">
          <ResultsTable />
        </div>
        {selectedMolecule && (
          <div className="w-96 border-l border-gray-700 overflow-y-auto">
            <MoleculeDetail molecule={selectedMolecule} />
          </div>
        )}
      </div>
      <StatusBar />
    </div>
  );
}

export default App;
