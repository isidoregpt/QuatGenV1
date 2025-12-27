import { create } from 'zustand';
import type { 
  Molecule, 
  MoleculeFilters, 
  GenerationStatus, 
  ObjectiveWeights,
  GenerationConstraints 
} from '../types';

interface AppState {
  // Connection
  isConnected: boolean;
  setConnected: (connected: boolean) => void;

  // Molecules
  molecules: Molecule[];
  setMolecules: (molecules: Molecule[]) => void;
  addMolecules: (molecules: Molecule[]) => void;
  totalMolecules: number;
  setTotalMolecules: (total: number) => void;

  // Selection
  selectedMolecule: Molecule | null;
  setSelectedMolecule: (molecule: Molecule | null) => void;
  selectedIds: Set<number>;
  toggleSelected: (id: number) => void;
  selectAll: () => void;
  clearSelection: () => void;

  // Filters
  filters: MoleculeFilters;
  setFilters: (filters: Partial<MoleculeFilters>) => void;

  // Generation
  generationStatus: GenerationStatus | null;
  setGenerationStatus: (status: GenerationStatus | null) => void;
  weights: ObjectiveWeights;
  setWeights: (weights: ObjectiveWeights) => void;
  constraints: GenerationConstraints;
  setConstraints: (constraints: Partial<GenerationConstraints>) => void;

  // UI
  showExportModal: boolean;
  setShowExportModal: (show: boolean) => void;
}

const defaultWeights: ObjectiveWeights = {
  efficacy: 0.4,
  safety: 0.3,
  environmental: 0.2,
  sa_score: 0.1,
};

const defaultConstraints: GenerationConstraints = {
  min_mw: 200,
  max_mw: 600,
  min_chain_length: 8,
  max_chain_length: 18,
  require_quat: true,
  require_novel: true,
  allowed_counterions: ['Cl', 'Br', 'I'],
};

const defaultFilters: MoleculeFilters = {
  pareto_only: false,
  starred_only: false,
};

export const useStore = create<AppState>((set, get) => ({
  // Connection
  isConnected: false,
  setConnected: (connected) => set({ isConnected: connected }),

  // Molecules
  molecules: [],
  setMolecules: (molecules) => set({ molecules }),
  addMolecules: (newMolecules) => set((state) => ({
    molecules: [...state.molecules, ...newMolecules]
  })),
  totalMolecules: 0,
  setTotalMolecules: (total) => set({ totalMolecules: total }),

  // Selection
  selectedMolecule: null,
  setSelectedMolecule: (molecule) => set({ selectedMolecule: molecule }),
  selectedIds: new Set(),
  toggleSelected: (id) => set((state) => {
    const newSet = new Set(state.selectedIds);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    return { selectedIds: newSet };
  }),
  selectAll: () => set((state) => ({
    selectedIds: new Set(state.molecules.map(m => m.id))
  })),
  clearSelection: () => set({ selectedIds: new Set() }),

  // Filters
  filters: defaultFilters,
  setFilters: (filters) => set((state) => ({
    filters: { ...state.filters, ...filters }
  })),

  // Generation
  generationStatus: null,
  setGenerationStatus: (status) => set({ generationStatus: status }),
  weights: defaultWeights,
  setWeights: (weights) => set({ weights }),
  constraints: defaultConstraints,
  setConstraints: (constraints) => set((state) => ({
    constraints: { ...state.constraints, ...constraints }
  })),

  // UI
  showExportModal: false,
  setShowExportModal: (show) => set({ showExportModal: show }),
}));
