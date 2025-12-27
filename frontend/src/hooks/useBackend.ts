import { useCallback } from 'react';
import axios from 'axios';
import { useStore } from './useStore';
import type { 
  GenerationRequest, 
  MoleculeListResponse, 
  ExportRequest 
} from '../types';

const API_BASE = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export function useBackend() {
  const { 
    setConnected, 
    setMolecules, 
    setTotalMolecules,
    setGenerationStatus,
    filters 
  } = useStore();

  const checkConnection = useCallback(async () => {
    try {
      await api.get('/status/system');
      setConnected(true);
      return true;
    } catch {
      setConnected(false);
      return false;
    }
  }, [setConnected]);

  const fetchMolecules = useCallback(async (
    limit = 100,
    offset = 0
  ) => {
    try {
      const params = new URLSearchParams({
        limit: limit.toString(),
        offset: offset.toString(),
        pareto_only: filters.pareto_only.toString(),
        starred_only: filters.starred_only.toString(),
      });
      
      if (filters.min_efficacy) {
        params.set('min_efficacy', filters.min_efficacy.toString());
      }
      
      const response = await api.get<MoleculeListResponse>(
        `/molecules?${params}`
      );
      
      setMolecules(response.data.molecules);
      setTotalMolecules(response.data.total);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch molecules:', error);
      throw error;
    }
  }, [filters, setMolecules, setTotalMolecules]);

  const startGeneration = useCallback(async (request: GenerationRequest) => {
    try {
      const response = await api.post('/generator/start', request);
      return response.data;
    } catch (error) {
      console.error('Failed to start generation:', error);
      throw error;
    }
  }, []);

  const stopGeneration = useCallback(async () => {
    try {
      const response = await api.post('/generator/stop');
      return response.data;
    } catch (error) {
      console.error('Failed to stop generation:', error);
      throw error;
    }
  }, []);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await api.get('/generator/status');
      setGenerationStatus(response.data);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch status:', error);
      throw error;
    }
  }, [setGenerationStatus]);

  const toggleStar = useCallback(async (id: number, starred: boolean) => {
    try {
      await api.patch(`/molecules/${id}`, { is_starred: starred });
    } catch (error) {
      console.error('Failed to toggle star:', error);
      throw error;
    }
  }, []);

  const exportMolecules = useCallback(async (request: ExportRequest) => {
    try {
      const response = await api.post(`/export/${request.format}`, request, {
        responseType: 'blob',
      });
      
      const blob = new Blob([response.data]);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `quat_molecules.${request.format}`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to export:', error);
      throw error;
    }
  }, []);

  return {
    checkConnection,
    fetchMolecules,
    startGeneration,
    stopGeneration,
    fetchStatus,
    toggleStar,
    exportMolecules,
  };
}
