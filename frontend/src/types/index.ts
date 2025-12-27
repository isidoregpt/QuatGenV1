export interface Molecule {
  id: number;
  smiles: string;
  efficacy_score: number;
  safety_score: number;
  environmental_score: number;
  sa_score: number;
  combined_score: number;
  molecular_weight?: number;
  logp?: number;
  chain_length?: number;
  is_valid_quat: boolean;
  is_pareto: boolean;
  is_starred: boolean;
  created_at: string;
}

export interface GenerationConstraints {
  min_mw: number;
  max_mw: number;
  min_chain_length: number;
  max_chain_length: number;
  require_quat: boolean;
  require_novel: boolean;
  allowed_counterions: string[];
}

export interface ObjectiveWeights {
  efficacy: number;
  safety: number;
  environmental: number;
  sa_score: number;
}

export interface GenerationRequest {
  num_molecules: number;
  constraints: GenerationConstraints;
  weights: ObjectiveWeights;
  batch_size: number;
  use_gpu: boolean;
  num_workers: number;
}

export interface GenerationStatus {
  is_running: boolean;
  molecules_generated: number;
  molecules_per_hour: number;
  pareto_frontier_size: number;
  current_batch: number;
  total_batches: number;
  elapsed_seconds: number;
  estimated_remaining_seconds: number;
  top_scores: { efficacy: number; safety: number; environmental: number; combined: number; };
}

export interface MoleculeListResponse {
  molecules: Molecule[];
  total: number;
  offset: number;
  limit: number;
}

export interface MoleculeFilters {
  pareto_only: boolean;
  starred_only: boolean;
  min_efficacy?: number;
}

export type ExportFormat = 'csv' | 'sdf' | 'pdf';

export interface ExportRequest {
  molecule_ids?: number[];
  format: ExportFormat;
  include_structures: boolean;
  include_properties: boolean;
  include_scores: boolean;
  pareto_only: boolean;
  starred_only: boolean;
}
