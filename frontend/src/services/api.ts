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

export default moleculeApi;
