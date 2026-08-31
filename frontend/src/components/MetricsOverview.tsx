import React from 'react';
import { Target } from 'lucide-react';
import type { DiseaseClassInfo } from '../types';

interface Props {
  classes: DiseaseClassInfo[];
}

export const MetricsOverview: React.FC<Props> = ({ classes }) => {
  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <h3 className="font-bold text-slate-100 text-sm uppercase tracking-wider mb-4 flex items-center gap-2">
        <Target className="w-4 h-4 text-teal-400" />
        Architecture & Disease Taxonomy
      </h3>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
        <div className="bg-slate-800/50 p-3 rounded-xl border border-slate-700/50">
          <span className="text-[11px] text-slate-400 block">Backbone</span>
          <span className="text-sm font-bold text-slate-200">MobileNetV2</span>
        </div>
        <div className="bg-slate-800/50 p-3 rounded-xl border border-slate-700/50">
          <span className="text-[11px] text-slate-400 block">Target Classes</span>
          <span className="text-sm font-bold text-teal-400">{classes.length || 7} Classes</span>
        </div>
        <div className="bg-slate-800/50 p-3 rounded-xl border border-slate-700/50">
          <span className="text-[11px] text-slate-400 block">Explainability</span>
          <span className="text-sm font-bold text-slate-200">Grad-CAM</span>
        </div>
        <div className="bg-slate-800/50 p-3 rounded-xl border border-slate-700/50">
          <span className="text-[11px] text-slate-400 block">Input Resolution</span>
          <span className="text-sm font-bold text-slate-200">224 × 224</span>
        </div>
      </div>

      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Class Directory ({classes.length})
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-56 overflow-y-auto pr-1">
          {classes.map((c) => (
            <div
              key={c.code}
              className="p-2.5 rounded-lg bg-slate-800/40 border border-slate-700/40 text-xs"
            >
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-200">{c.name}</span>
                <span className="font-mono text-[10px] text-teal-400 bg-teal-500/10 px-1.5 py-0.5 rounded border border-teal-500/20">
                  {c.code}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1 line-clamp-1">{c.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
