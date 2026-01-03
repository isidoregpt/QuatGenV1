/**
 * API service for molecule image rendering and data fetching
 */

const API_BASE = 'http://localhost:8000/api';

export interface RenderOptions {
  width?: number;
  height?: number;
  format?: 'png' | 'svg';
  highlightQuat?: boolean;
  highlightSmarts?: string;
}

export interface MoleculeImageResponse {
  smiles: string;
  image_data_uri: string;
  molecule_info?: {
    num_atoms: number;
    num_heavy_atoms: number;
    formula: string;
    has_quat_nitrogen: boolean;
    quat_nitrogen_count: number;
  };
}

export interface GridImageResponse {
  num_molecules: number;
  image_base64: string;
  mime_type: string;
}

export interface ComparisonImageResponse {
  image_base64: string;
  mime_type: string;
}

export const moleculeApi = {
  /**
   * Get molecule image by ID
   */
  getMoleculeImage: async (
    moleculeId: number,
    options: RenderOptions = {}
  ): Promise<string> => {
    const params = new URLSearchParams();
    if (options.width) params.append('width', options.width.toString());
    if (options.height) params.append('height', options.height.toString());
    if (options.format) params.append('format', options.format);
    if (options.highlightQuat !== undefined) {
      params.append('highlight_quat', options.highlightQuat.toString());
    }

    const response = await fetch(
      `${API_BASE}/molecules/${moleculeId}/image/base64?${params}`
    );
    if (!response.ok) throw new Error('Failed to fetch molecule image');
    const data = await response.json();
    return data.image_data_uri;
  },

  /**
   * Render any SMILES to image
   */
  renderSmiles: async (
    smiles: string,
    options: RenderOptions = {}
  ): Promise<MoleculeImageResponse> => {
    const params = new URLSearchParams({ smiles });
    if (options.width) params.append('width', options.width.toString());
    if (options.height) params.append('height', options.height.toString());
    if (options.format) params.append('format', options.format);
    if (options.highlightQuat !== undefined) {
      params.append('highlight_quat', options.highlightQuat.toString());
    }
    if (options.highlightSmarts) {
      params.append('highlight_smarts', options.highlightSmarts);
    }

    const response = await fetch(
      `${API_BASE}/molecules/render?${params}`,
      { method: 'POST' }
    );
    if (!response.ok) throw new Error('Failed to render molecule');
    return response.json();
  },

  /**
   * Render grid of molecules
   */
  renderGrid: async (
    smilesList: string[],
    legends?: string[],
    options: RenderOptions = {}
  ): Promise<GridImageResponse> => {
    const response = await fetch(`${API_BASE}/molecules/render/grid`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        smiles_list: smilesList,
        legends,
        width: options.width || 800,
        height: options.height || 600,
        mols_per_row: 4,
        format: options.format || 'png'
      })
    });
    if (!response.ok) throw new Error('Failed to render grid');
    return response.json();
  },

  /**
   * Render comparison of two molecules
   */
  renderComparison: async (
    smiles1: string,
    smiles2: string,
    label1?: string,
    label2?: string
  ): Promise<ComparisonImageResponse> => {
    const response = await fetch(`${API_BASE}/molecules/render/comparison`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        smiles1,
        smiles2,
        label1: label1 || 'Molecule 1',
        label2: label2 || 'Molecule 2',
        width: 800,
        height: 400
      })
    });
    if (!response.ok) throw new Error('Failed to render comparison');
    return response.json();
  },

  /**
   * Get molecule data by ID
   */
  getMolecule: async (moleculeId: number): Promise<any> => {
    const response = await fetch(`${API_BASE}/molecules/${moleculeId}`);
    if (!response.ok) throw new Error('Failed to fetch molecule');
    return response.json();
  },

  /**
   * Get renderer status
   */
  getRendererStatus: async (): Promise<any> => {
    const response = await fetch(`${API_BASE}/molecules/render/status`);
    if (!response.ok) throw new Error('Failed to fetch renderer status');
    return response.json();
  }
};

// Search API types and methods

export interface SearchPattern {
  name: string;
  smarts: string;
  description: string;
  category: string;
}

export interface SubstructureSearchRequest {
  smarts: string;
  max_results?: number;
  require_quat?: boolean;
  min_efficacy?: number;
  min_safety?: number;
}

export interface SimilaritySearchRequest {
  smiles: string;
  threshold?: number;
  max_results?: number;
}

export interface SearchResult {
  smiles: string;
  molecule_id: number | null;
  name?: string;
  match_count: number;
  match_atoms: number[][];
  scores: {
    efficacy?: number;
    safety?: number;
    sa?: number;
  };
  metadata?: Record<string, any>;
}

export interface SimilarityResult {
  smiles: string;
  molecule_id: number | null;
  name?: string;
  similarity: number;
  scores: {
    efficacy_score?: number;
    safety_score?: number;
    sa_score?: number;
  };
}

export interface SubstructureSearchResponse {
  query: string;
  results: SearchResult[];
  total_searched: number;
}

export interface SimilaritySearchResponse {
  query: string;
  threshold: number;
  results: SimilarityResult[];
}

export interface PatternsResponse {
  all_patterns: Record<string, string>;
  quat_patterns: Record<string, string>;
}

export const searchApi = {
  /**
   * Get available SMARTS patterns
   */
  getPatterns: async (): Promise<PatternsResponse> => {
    const response = await fetch(`${API_BASE}/search/patterns`);
    if (!response.ok) throw new Error('Failed to fetch patterns');
    return response.json();
  },

  /**
   * Validate SMARTS pattern
   */
  validateSmarts: async (smarts: string): Promise<{ smarts: string; is_valid: boolean; error: string | null }> => {
    const params = new URLSearchParams({ smarts });
    const response = await fetch(`${API_BASE}/search/validate?${params}`, {
      method: 'POST'
    });
    if (!response.ok) throw new Error('Failed to validate SMARTS');
    return response.json();
  },

  /**
   * Substructure search
   */
  substructureSearch: async (request: SubstructureSearchRequest): Promise<SearchResult[]> => {
    const response = await fetch(`${API_BASE}/search/substructure`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    if (!response.ok) throw new Error('Search failed');
    return response.json();
  },

  /**
   * Similarity search
   */
  similaritySearch: async (request: SimilaritySearchRequest): Promise<SimilarityResult[]> => {
    const response = await fetch(`${API_BASE}/search/similarity`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    if (!response.ok) throw new Error('Search failed');
    return response.json();
  },

  /**
   * Classify quat type
   */
  classifyQuat: async (smiles: string): Promise<{
    smiles: string;
    is_quat: boolean;
    quat_type: string | null;
  }> => {
    const params = new URLSearchParams({ smiles });
    const response = await fetch(`${API_BASE}/search/classify?${params}`, {
      method: 'POST'
    });
    if (!response.ok) throw new Error('Classification failed');
    return response.json();
  },

  /**
   * Get structural features
   */
  getFeatures: async (smiles: string): Promise<{
    smiles: string;
    features: Record<string, boolean | string | null>;
  }> => {
    const params = new URLSearchParams({ smiles });
    const response = await fetch(`${API_BASE}/search/features?${params}`, {
      method: 'POST'
    });
    if (!response.ok) throw new Error('Failed to get features');
    return response.json();
  },

  /**
   * Check if molecule has quat nitrogen
   */
  checkHasQuat: async (smiles: string): Promise<{
    smiles: string;
    has_quat_nitrogen: boolean;
  }> => {
    const params = new URLSearchParams({ smiles });
    const response = await fetch(`${API_BASE}/search/quat-check?${params}`);
    if (!response.ok) throw new Error('Check failed');
    return response.json();
  },

  /**
   * Get search status
   */
  getStatus: async (): Promise<{
    available: boolean;
    features: Record<string, boolean>;
    common_patterns: number;
    quat_patterns: number;
  }> => {
    const response = await fetch(`${API_BASE}/search/status`);
    if (!response.ok) throw new Error('Failed to fetch status');
    return response.json();
  }
};

// Benchmark API types and methods

export interface PropertyComparison {
  property: string;
  generated: number;
  reference: number;
  outcome: 'better' | 'similar' | 'worse' | 'unknown';
  interpretation: string;
}

export interface BenchmarkResult {
  smiles: string;
  molecule_id?: number;
  overall_score: number;
  recommendation: string;
  confidence: number;
  scaffold_type: string;
  structural_novelty: number;
  closest_references: Array<{
    name: string;
    similarity: number;
    smiles?: string;
    mic_s_aureus?: number;
    mic_e_coli?: number;
    ld50?: number;
    applications?: string[];
  }>;
  property_comparisons: PropertyComparison[];
  advantages: string[];
  disadvantages: string[];
  properties_better: number;
  properties_similar: number;
  properties_worse: number;
}

export interface BenchmarkReportSummary {
  generated_at: string;
  summary: {
    total_molecules: number;
    molecules_benchmarked: number;
    avg_overall_score: number;
    top_candidates_count: number;
  };
  scaffold_distribution: Record<string, number>;
  top_candidates: Array<{
    smiles: string;
    molecule_id?: number;
    overall_score: number;
    recommendation: string;
    scaffold_type: string;
    structural_novelty: number;
    advantages: string[];
    closest_reference?: string;
  }>;
  reference_comparison: {
    total_comparisons: number;
    better_than_reference: number;
    similar_to_reference: number;
    worse_than_reference: number;
    closest_references_used: Record<string, number>;
    property_improvements: Record<string, number>;
    property_deficits: Record<string, number>;
  };
  recommendations: string[];
}

export interface ReferenceCompound {
  name: string;
  smiles: string;
  chembl_id?: string;
  mic_s_aureus?: number;
  mic_e_coli?: number;
  mic_p_aeruginosa?: number;
  mic_c_albicans?: number;
  ld50_oral_rat?: number;
  applications?: string[];
}

export const benchmarkApi = {
  /**
   * Get benchmark service status
   */
  getStatus: async (): Promise<{
    available: boolean;
    reference_compounds: number;
    features: string[];
  }> => {
    const response = await fetch(`${API_BASE}/benchmark/status`);
    if (!response.ok) throw new Error('Failed to fetch benchmark status');
    return response.json();
  },

  /**
   * Benchmark a single molecule by SMILES
   */
  benchmarkMolecule: async (
    smiles: string,
    predictedScores?: Record<string, number>
  ): Promise<BenchmarkResult> => {
    const response = await fetch(`${API_BASE}/benchmark/molecule`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles, predicted_scores: predictedScores })
    });
    if (!response.ok) throw new Error('Benchmark failed');
    return response.json();
  },

  /**
   * Benchmark molecule by database ID
   */
  benchmarkMoleculeById: async (moleculeId: number): Promise<BenchmarkResult> => {
    const response = await fetch(`${API_BASE}/benchmark/molecule/${moleculeId}`);
    if (!response.ok) throw new Error('Benchmark failed');
    return response.json();
  },

  /**
   * Batch benchmark molecules
   */
  benchmarkBatch: async (
    moleculeIds?: number[],
    topN: number = 20,
    minScore: number = 0
  ): Promise<{
    count: number;
    molecules_analyzed: number;
    results: BenchmarkResult[];
  }> => {
    const response = await fetch(`${API_BASE}/benchmark/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        molecule_ids: moleculeIds,
        top_n: topN,
        min_score: minScore
      })
    });
    if (!response.ok) throw new Error('Batch benchmark failed');
    return response.json();
  },

  /**
   * Generate comprehensive benchmark report
   */
  generateReport: async (moleculeIds?: number[]): Promise<BenchmarkReportSummary> => {
    const response = await fetch(`${API_BASE}/benchmark/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ molecule_ids: moleculeIds })
    });
    if (!response.ok) throw new Error('Report generation failed');
    return response.json();
  },

  /**
   * Get reference compounds
   */
  getReferences: async (): Promise<{
    count: number;
    compounds: ReferenceCompound[];
  }> => {
    const response = await fetch(`${API_BASE}/benchmark/references`);
    if (!response.ok) throw new Error('Failed to fetch references');
    return response.json();
  },

  /**
   * Get benchmark criteria and thresholds
   */
  getCriteria: async (): Promise<{
    property_thresholds: Record<string, number>;
    optimization_directions: Record<string, boolean | null>;
    score_interpretation: Record<string, string>;
    scaffold_types: string[];
  }> => {
    const response = await fetch(`${API_BASE}/benchmark/criteria`);
    if (!response.ok) throw new Error('Failed to fetch criteria');
    return response.json();
  }
};

export default moleculeApi;
