import React from 'react';
import { Zap, ShieldAlert, AlertTriangle } from 'lucide-react';
import type { PredictionResponse } from '../types';

interface Props {
  prediction: PredictionResponse;
  onSelectTargetClass?: (code: string) => void;
  activeTargetClass?: string | null;
}

export const PredictionCard: React.FC<Props> = ({
  prediction,
  onSelectTargetClass,
  activeTargetClass,
}) => {
  const getSeverityBadge = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'high':
        return 'bg-rose-500/20 text-rose-300 border-rose-500/30';
      case 'moderate':
      case 'precancerous':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/30';
      default:
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30';
    }
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
      {/* Primary Output Header */}
      <div className="flex items-start justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
            Primary Diagnostic Output
          </span>
          <h2 className="text-xl font-bold text-slate-100 mt-0.5">
            {prediction.predicted_name}
          </h2>
          <p className="text-xs text-slate-400 mt-1 line-clamp-2">
            {prediction.description}
          </p>
        </div>
        <div className="text-right flex flex-col items-end flex-shrink-0">
          <span className={`text-xs px-2.5 py-1 rounded-full font-medium border ${getSeverityBadge(prediction.severity)}`}>
            {prediction.severity}
          </span>
          <div className="mt-2 text-2xl font-black text-teal-400">
            {prediction.percentage}%
          </div>
          <span className="text-[11px] text-slate-400">Confidence</span>
        </div>
      </div>

      {/* Multiclass Probability Bars */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Multiclass Probability Distribution
          </h4>
          <span className="text-[10px] text-slate-400">
            Click class to inspect Grad-CAM
          </span>
        </div>

        <div className="space-y-2">
          {prediction.top_predictions.map((item) => {
            const isTarget = (activeTargetClass || prediction.predicted_code) === item.code;
            const isPrimary = item.code === prediction.predicted_code;

            return (
              <button
                key={item.code}
                type="button"
                onClick={() => onSelectTargetClass?.(item.code)}
                className={`w-full text-left p-1.5 rounded-lg transition-colors group ${
                  isTarget ? 'bg-slate-800/90 ring-1 ring-teal-500/50' : 'hover:bg-slate-800/40'
                }`}
              >
                <div className="flex justify-between items-center text-xs mb-1">
                  <span className={`font-medium flex items-center gap-1.5 ${isPrimary ? 'text-teal-300 font-semibold' : 'text-slate-300'}`}>
                    {item.name}
                    {isPrimary && (
                      <span className="text-[10px] px-1.5 py-0.2 rounded bg-teal-500/20 text-teal-300 border border-teal-500/30">
                        Top
                      </span>
                    )}
                  </span>
                  <span className="text-slate-400 font-mono text-[11px]">{item.percentage}%</span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      isPrimary ? 'bg-teal-400' : 'bg-slate-600'
                    }`}
                    style={{ width: `${Math.max(item.percentage, 1)}%` }}
                  />
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Clinical Disclaimer Banner */}
      <div className="p-3 rounded-xl bg-amber-950/30 border border-amber-800/40 text-[11px] text-amber-300/90 flex items-start gap-2.5">
        <AlertTriangle className="w-4 h-4 flex-shrink-0 text-amber-400 mt-0.5" />
        <span>{prediction.disclaimer}</span>
      </div>

      {/* Metadata Footer */}
      <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400">
        <span className="flex items-center gap-1">
          <Zap className="w-3.5 h-3.5 text-amber-400" />
          Inference Latency: <strong className="text-slate-200">{prediction.inference_time_ms} ms</strong>
        </span>
        <span className="flex items-center gap-1">
          <ShieldAlert className="w-3.5 h-3.5 text-teal-400" />
          Device: <strong className="text-slate-200 uppercase">{prediction.device}</strong>
        </span>
      </div>
    </div>
  );
};
