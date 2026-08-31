import axios from 'axios';
import type { HealthStatus, PredictionResponse, DiseaseClassInfo } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';

export const fetchHealth = async (): Promise<HealthStatus> => {
  const res = await axios.get<HealthStatus>(`${API_BASE}/health`, { timeout: 10000 });
  return res.data;
};

export const fetchClasses = async (): Promise<DiseaseClassInfo[]> => {
  const res = await axios.get<DiseaseClassInfo[]>(`${API_BASE}/classes`, { timeout: 10000 });
  return res.data;
};

export const analyzeSkinImage = async (
  file: File,
  targetClass?: string,
  includeGradcam: boolean = true
): Promise<PredictionResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  if (targetClass) {
    formData.append('target_class', targetClass);
  }
  formData.append('include_gradcam', includeGradcam ? 'true' : 'false');

  const res = await axios.post<PredictionResponse>(`${API_BASE}/predict`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 30000, // 30s timeout for model inference
  });
  return res.data;
};
