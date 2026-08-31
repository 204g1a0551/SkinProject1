import { useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { ImageUploader } from './components/ImageUploader';
import { PredictionCard } from './components/PredictionCard';
import { GradcamViewer } from './components/GradcamViewer';
import { MetricsOverview } from './components/MetricsOverview';
import { fetchHealth, fetchClasses, analyzeSkinImage } from './services/api';
import type { HealthStatus, PredictionResponse, DiseaseClassInfo } from './types';
import { Loader2, Sparkles, AlertCircle, RefreshCw, RotateCcw } from 'lucide-react';

export function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [classes, setClasses] = useState<DiseaseClassInfo[]>([]);
  const [loadingInitial, setLoadingInitial] = useState(true);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [activeTargetClass, setActiveTargetClass] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const initData = async () => {
    setLoadingInitial(true);
    setError(null);
    try {
      const [h, c] = await Promise.all([fetchHealth(), fetchClasses()]);
      setHealth(h);
      setClasses(c);
    } catch (err: any) {
      console.error(err);
      setError('Could not connect to FastAPI backend server. Ensure uvicorn is running on port 8000.');
    } finally {
      setLoadingInitial(false);
    }
  };

  useEffect(() => {
    initData();
  }, []);

  const handleImageSelected = (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setPrediction(null);
    setActiveTargetClass(null);
    setError(null);
  };

  const handleRunInference = async (targetClassOverride?: string) => {
    if (!selectedFile) return;
    setAnalyzing(true);
    setError(null);
    try {
      const target = targetClassOverride || activeTargetClass || undefined;
      const res = await analyzeSkinImage(selectedFile, target);
      setPrediction(res);
      if (target) {
        setActiveTargetClass(target);
      }
    } catch (err: any) {
      console.error(err);
      setError(
        err?.response?.data?.detail ||
          'Inference connection failure. Please verify the backend is active.'
      );
    } finally {
      setAnalyzing(false);
    }
  };

  const handleSelectTargetClass = (classCode: string) => {
    setActiveTargetClass(classCode);
    handleRunInference(classCode);
  };

  const handleReset = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setPrediction(null);
    setActiveTargetClass(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Navbar health={health} loading={loadingInitial} />

      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
        {/* Error Banner with Retry */}
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-300 text-sm flex items-center justify-between shadow-lg">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-5 h-5 flex-shrink-0 text-rose-400" />
              <span>{error}</span>
            </div>
            <button
              onClick={initData}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-rose-900/60 hover:bg-rose-800 rounded-lg text-xs font-medium transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" /> Retry Connection
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Input, Upload, and Actions */}
          <div className="lg:col-span-5 space-y-6">
            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl">
              <div className="flex items-center justify-between mb-1">
                <h2 className="text-base font-bold text-slate-100">
                  Skin Lesion Image Input
                </h2>
                {selectedFile && (
                  <button
                    onClick={handleReset}
                    className="flex items-center gap-1 text-xs text-slate-400 hover:text-rose-400 transition-colors"
                    title="Clear selected image"
                  >
                    <RotateCcw className="w-3.5 h-3.5" /> Clear / Reset
                  </button>
                )}
              </div>
              <p className="text-xs text-slate-400 mb-4">
                Upload a dermatoscopic photograph for automated MobileNetV2 classification and Grad-CAM interpretability.
              </p>

              <ImageUploader
                onImageSelected={handleImageSelected}
                isLoading={analyzing}
                selectedPreview={previewUrl}
              />

              <div className="mt-5 flex gap-3">
                <button
                  onClick={() => handleRunInference()}
                  disabled={!selectedFile || analyzing}
                  className={`flex-1 py-3 px-4 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all shadow-lg ${
                    !selectedFile || analyzing
                      ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                      : 'bg-teal-500 hover:bg-teal-400 text-slate-950 shadow-teal-500/20 active:scale-[0.99]'
                  }`}
                >
                  {analyzing ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Analyzing Lesion & Generating Grad-CAM...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Run AI Diagnostic & Grad-CAM
                    </>
                  )}
                </button>

                {prediction && (
                  <button
                    onClick={handleReset}
                    disabled={analyzing}
                    className="px-4 py-3 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 rounded-xl text-sm font-medium transition-colors flex items-center gap-1.5"
                    title="Start new analysis"
                  >
                    <RotateCcw className="w-4 h-4" /> Reset
                  </button>
                )}
              </div>
            </div>

            <MetricsOverview classes={classes} />
          </div>

          {/* Right Column: Prediction Results & Grad-CAM Visualizations */}
          <div className="lg:col-span-7 space-y-6">
            {prediction ? (
              <>
                <PredictionCard
                  prediction={prediction}
                  onSelectTargetClass={handleSelectTargetClass}
                  activeTargetClass={activeTargetClass}
                />
                <GradcamViewer
                  originalImage={previewUrl}
                  gradcamBase64={prediction.gradcam_base64}
                  predictedName={prediction.predicted_name}
                  targetLayer={prediction.target_layer}
                  explainedClassCode={prediction.explained_class_code}
                />
              </>
            ) : (
              <div className="bg-slate-900/40 border border-dashed border-slate-800 rounded-2xl p-12 flex flex-col items-center justify-center text-center min-h-[480px]">
                <div className="w-16 h-16 rounded-2xl bg-slate-800/60 border border-slate-700 text-slate-500 flex items-center justify-center mb-4">
                  <Sparkles className="w-8 h-8 text-teal-400/50" />
                </div>
                <h3 className="text-base font-semibold text-slate-300 mb-1">
                  Ready for Diagnostic Analysis
                </h3>
                <p className="text-xs text-slate-500 max-w-sm">
                  Upload a dermatoscopic lesion photograph on the left and click "Run AI Diagnostic & Grad-CAM" to generate predictions and attention heatmaps.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="border-t border-slate-800/80 bg-slate-900/40 py-4 text-center text-xs text-slate-500">
        AI Skin: Intelligent Skin Diseases Detection • Final-Year Academic Project • MobileNetV2 + Grad-CAM Explainable AI
      </footer>
    </div>
  );
}

export default App;
