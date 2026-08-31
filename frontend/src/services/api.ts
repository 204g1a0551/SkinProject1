import axios from 'axios';
import type { HealthStatus, PredictionResponse, DiseaseClassInfo } from '../types';

// Ensure no trailing slash and append /api if not already included
let rawBase = import.meta.env.VITE_API_BASE || '/api';
if (rawBase.endsWith('/')) {
  rawBase = rawBase.slice(0, -1);
}
const API_BASE = rawBase.endsWith('/api') ? rawBase : `${rawBase}/api`;

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
    timeout: 45000, // 45s timeout for model inference
  });
  return res.data;
};
