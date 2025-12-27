import { useCallback } from 'react';
import axios from 'axios';
import { useStore } from './useStore';
import type { GenerationRequest, MoleculeListResponse } from '../types';

const api = axios.create({ baseURL: 'http://localhost:8000/api', timeout: 30000 });

export function useBackend() {
  const { setConnected, setMolecules, setTotalMolecules, setGenerationStatus, filters } = useStore();

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

  const fetchMolecules = useCallback(async (limit = 100, offset = 0) => {
    try {
      const params = new URLSearchParams({ limit: limit.toString(), offset: offset.toString(),
        pareto_only: filters.pareto_only.toString(), starred_only: filters.starred_only.toString() });
      const response = await api.get<MoleculeListResponse>(`/molecules?${params}`);
      setMolecules(response.data.molecules);
      setTotalMolecules(response.data.total);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch molecules:', error);
      throw error;
    }
  }, [filters, setMolecules, setTotalMolecules]);

  const startGeneration = useCallback(async (request: GenerationRequest) => {
    const response = await api.post('/generator/start', request);
    return response.data;
  }, []);

  const stopGeneration = useCallback(async () => {
    const response = await api.post('/generator/stop');
    return response.data;
  }, []);

  const fetchStatus = useCallback(async () => {
    const response = await api.get('/generator/status');
    setGenerationStatus(response.data);
    return response.data;
  }, [setGenerationStatus]);

  const toggleStar = useCallback(async (id: number, starred: boolean) => {
    await api.patch(`/molecules/${id}`, { is_starred: starred });
  }, []);

  return { checkConnection, fetchMolecules, startGeneration, stopGeneration, fetchStatus, toggleStar };
}
