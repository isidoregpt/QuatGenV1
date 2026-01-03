import React, { useEffect, useState } from 'react';
import { Search, Beaker, List } from 'lucide-react';
import { GeneratorControls } from './components/GeneratorControls';
import { ResultsTable } from './components/ResultsTable';
import { MoleculeDetail } from './components/MoleculeDetail';
import { StatusBar } from './components/StatusBar';
import { SearchPanel } from './components/search';
import { useStore } from './hooks/useStore';
import { useBackend } from './hooks/useBackend';

type SidebarView = 'generator' | 'search';

function App() {
  const { selectedMolecule, setSelectedMolecule } = useStore();
  const { checkConnection, fetchMolecules } = useBackend();
  const [sidebarView, setSidebarView] = useState<SidebarView>('generator');

  useEffect(() => { checkConnection(); }, []);

  const handleSearchSelectMolecule = async (moleculeId: number) => {
    // Fetch molecule details and set as selected
    try {
      const response = await fetch(`http://localhost:8000/api/molecules/${moleculeId}`);
      if (response.ok) {
        const molecule = await response.json();
        setSelectedMolecule(molecule);
      }
    } catch (err) {
      console.error('Failed to fetch molecule:', err);
    }
  };

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-gray-100">
      <header className="bg-gray-800 border-b border-gray-700 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">Q</span>
            </div>
            <h1 className="text-xl font-semibold">Quat Generator Pro</h1>
          </div>

          {/* View Toggle */}
          <div className="flex gap-1 bg-gray-700 rounded-lg p-1">
            <button
              onClick={() => setSidebarView('generator')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                sidebarView === 'generator'
                  ? 'bg-emerald-600 text-white'
                  : 'text-gray-300 hover:text-white hover:bg-gray-600'
              }`}
            >
              <Beaker size={16} />
              Generate
            </button>
            <button
              onClick={() => setSidebarView('search')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                sidebarView === 'search'
                  ? 'bg-emerald-600 text-white'
                  : 'text-gray-300 hover:text-white hover:bg-gray-600'
              }`}
            >
              <Search size={16} />
              Search
            </button>
          </div>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <div className="w-96 border-r border-gray-700 overflow-y-auto">
          {sidebarView === 'generator' ? (
            <GeneratorControls />
          ) : (
            <div className="p-4">
              <SearchPanel
                onSelectMolecule={handleSearchSelectMolecule}
              />
            </div>
          )}
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <ResultsTable />
        </div>

        {/* Molecule Detail Panel */}
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
