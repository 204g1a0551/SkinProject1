import React, { useState } from 'react';
import { Layers, Sparkles, Eye, Columns } from 'lucide-react';

interface Props {
  originalImage: string | null;
  gradcamBase64: string | null;
  predictedName: string;
  targetLayer?: string;
  explainedClassCode?: string;
}

export const GradcamViewer: React.FC<Props> = ({
  originalImage,
  gradcamBase64,
  predictedName,
  targetLayer = 'features.18.0',
  explainedClassCode,
}) => {
  const [viewMode, setViewMode] = useState<'side-by-side' | 'gradcam-only' | 'original-only'>('side-by-side');

  if (!gradcamBase64 || !originalImage) {
    return null;
  }

  // Ensure data URI prefix
  const gradcamUrl = gradcamBase64.startsWith('data:')
    ? gradcamBase64
    : `data:image/jpeg;base64,${gradcamBase64}`;

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <h3 className="font-bold text-slate-100 flex items-center gap-2 text-base">
            <Sparkles className="w-4 h-4 text-teal-400" />
            Grad-CAM Visual Saliency Heatmap
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Model attention gradients highlighting morphological patterns for{' '}
            <strong className="text-slate-200">{predictedName}</strong>
            {explainedClassCode && (
              <span className="ml-1 text-[11px] text-teal-400 font-mono">({explainedClassCode})</span>
            )}
          </p>
        </div>

        {/* View Mode Switcher */}
        <div className="flex items-center gap-1 bg-slate-800/80 p-1 rounded-lg border border-slate-700 text-xs">
          <button
            type="button"
            onClick={() => setViewMode('side-by-side')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-md transition-colors ${
              viewMode === 'side-by-side'
                ? 'bg-teal-500 text-slate-950 font-semibold shadow-sm'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            <Columns className="w-3 h-3" /> Side by Side
          </button>
          <button
            type="button"
            onClick={() => setViewMode('gradcam-only')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-md transition-colors ${
              viewMode === 'gradcam-only'
                ? 'bg-teal-500 text-slate-950 font-semibold shadow-sm'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            <Sparkles className="w-3 h-3" /> Grad-CAM Overlay
          </button>
          <button
            type="button"
            onClick={() => setViewMode('original-only')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-md transition-colors ${
              viewMode === 'original-only'
                ? 'bg-teal-500 text-slate-950 font-semibold shadow-sm'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            <Eye className="w-3 h-3" /> Original
          </button>
        </div>
      </div>

      {/* Image Displays */}
      <div className="flex flex-wrap justify-center items-center gap-6 py-2">
        {(viewMode === 'side-by-side' || viewMode === 'original-only') && (
          <div className="flex flex-col items-center gap-2">
            <span className="text-xs font-medium text-slate-400">Original Dermatoscopy</span>
            <div className="w-64 h-64 rounded-xl overflow-hidden border border-slate-700 bg-slate-950 shadow-md">
              <img src={originalImage} alt="Original Lesion" className="w-full h-full object-contain" />
            </div>
          </div>
        )}

        {(viewMode === 'side-by-side' || viewMode === 'gradcam-only') && (
          <div className="flex flex-col items-center gap-2">
            <span className="text-xs font-medium text-teal-400">Grad-CAM Salience Overlay</span>
            <div className="w-64 h-64 rounded-xl overflow-hidden border border-teal-500/40 bg-slate-950 shadow-md">
              <img src={gradcamUrl} alt="Grad-CAM Heatmap" className="w-full h-full object-contain" />
            </div>
          </div>
        )}
      </div>

      {/* Technical Interpretability Footer */}
      <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700/60 text-xs text-slate-400 flex items-start gap-2">
        <Layers className="w-4 h-4 text-teal-400 flex-shrink-0 mt-0.5" />
        <span>
          <strong>Target Convolutional Layer:</strong> <code>{targetLayer}</code>. Warmer regions (red/yellow) indicate discriminative visual features positively driving the neural network's activation score.
        </span>
      </div>
    </div>
  );
};
