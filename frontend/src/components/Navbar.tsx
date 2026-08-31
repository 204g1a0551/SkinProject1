import React from 'react';
import { Activity, Cpu } from 'lucide-react';
import type { HealthStatus } from '../types';

interface Props {
  health: HealthStatus | null;
  loading: boolean;
}

export const Navbar: React.FC<Props> = ({ health, loading }) => {
  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-teal-500/10 text-teal-400 border border-teal-500/20 shadow-inner">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <h1 className="font-bold text-lg text-slate-100 tracking-tight flex items-center gap-2">
              AI Skin
              <span className="text-xs px-2 py-0.5 rounded-full bg-teal-500/20 text-teal-300 font-normal border border-teal-500/30">
                Phase 1 Scaffolding
              </span>
            </h1>
            <p className="text-xs text-slate-400 hidden sm:block">Intelligent Skin Diseases Detection • MobileNetV2 + Grad-CAM</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-xs">
            <Cpu className="w-4 h-4 text-indigo-400" />
            <span className="text-slate-400">Device:</span>
            <span className="font-medium text-slate-200 uppercase">
              {health ? health.device : loading ? 'Detecting...' : 'Offline'}
            </span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-xs">
            <span className={`w-2 h-2 rounded-full ${health?.status === 'healthy' ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
            <span className="text-slate-300 font-medium">
              {health?.status === 'healthy' ? 'Backend Live' : 'Backend Disconnected'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
