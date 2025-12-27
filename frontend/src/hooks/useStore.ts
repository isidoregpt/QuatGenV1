import { create } from 'zustand';
import type { Molecule, MoleculeFilters, GenerationStatus, ObjectiveWeights, GenerationConstraints } from '../types';

interface AppState {
  isConnected: boolean;
  setConnected: (connected: boolean) => void;
  molecules: Molecule[];
  setMolecules: (molecules: Molecule[]) => void;
  totalMolecules: number;
  setTotalMolecules: (total: number) => void;
  selectedMolecule: Molecule | null;
  setSelectedMolecule: (molecule: Molecule | null) => void;
  filters: MoleculeFilters;
  setFilters: (filters: Partial<MoleculeFilters>) => void;
  generationStatus: GenerationStatus | null;
  setGenerationStatus: (status: GenerationStatus | null) => void;
  weights: ObjectiveWeights;
  setWeights: (weights: ObjectiveWeights) => void;
  constraints: GenerationConstraints;
  setConstraints: (constraints: Partial<GenerationConstraints>) => void;
}

export const useStore = create<AppState>((set) => ({
  isConnected: false,
  setConnected: (connected) => set({ isConnected: connected }),
  molecules: [],
  setMolecules: (molecules) => set({ molecules }),
  totalMolecules: 0,
  setTotalMolecules: (total) => set({ totalMolecules: total }),
  selectedMolecule: null,
  setSelectedMolecule: (molecule) => set({ selectedMolecule: molecule }),
  filters: { pareto_only: false, starred_only: false },
  setFilters: (filters) => set((state) => ({ filters: { ...state.filters, ...filters } })),
  generationStatus: null,
  setGenerationStatus: (status) => set({ generationStatus: status }),
  weights: { efficacy: 0.4, safety: 0.3, environmental: 0.2, sa_score: 0.1 },
  setWeights: (weights) => set({ weights }),
  constraints: { min_mw: 200, max_mw: 600, min_chain_length: 8, max_chain_length: 18, require_quat: true, require_novel: true, allowed_counterions: ['Cl', 'Br', 'I'] },
  setConstraints: (constraints) => set((state) => ({ constraints: { ...state.constraints, ...constraints } }))
}));
