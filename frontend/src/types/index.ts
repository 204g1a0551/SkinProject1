export interface DiseaseClassInfo {
  code: string;
  name: string;
  severity: string;
  description: string;
}

export interface PredictionScore {
  code: string;
  name: string;
  confidence: number;
  percentage: number;
  severity: string;
}

export interface PredictionResponse {
  success: boolean;
  predicted_code: string;
  predicted_name: string;
  confidence: number;
  percentage: number;
  severity: string;
  description: string;
  top_predictions: PredictionScore[];
  gradcam_base64: string | null;
  target_layer?: string;
  explained_class_code?: string;
  inference_time_ms: number;
  device: string;
  disclaimer: string;
}

export interface HealthStatus {
  status: string;
  app_name: string;
  version: string;
  device: string;
  mps_available: boolean;
  cuda_available: boolean;
  model_loaded: boolean;
  weights_path?: string;
  num_classes: number;
}
