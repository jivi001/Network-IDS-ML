import React from 'react';
import { Activity, Database, Server, Shield } from 'lucide-react';

export default function SystemHealthWidget() {
  return (
    <div className="bg-[#111827]/80 backdrop-blur-xl border border-gray-800 rounded-2xl p-5 shadow-lg flex flex-col justify-between h-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-gray-300 font-semibold text-sm tracking-wide uppercase">System Health</h2>
        <span className="flex h-3 w-3 relative">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
          <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
            <Activity size={16} className="text-emerald-500" />
          </div>
          <div>
            <p className="text-xs text-gray-500">Tier 1 Engine</p>
            <p className="text-sm font-semibold text-gray-200">Online</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="p-2 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
            <Database size={16} className="text-emerald-500" />
          </div>
          <div>
            <p className="text-xs text-gray-500">Tier 2 VAE</p>
            <p className="text-sm font-semibold text-gray-200">Online</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="p-2 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
            <Server size={16} className="text-emerald-500" />
          </div>
          <div>
            <p className="text-xs text-gray-500">Data Stream</p>
            <p className="text-sm font-semibold text-gray-200">Active</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-500/10 rounded-lg border border-blue-500/20">
            <Shield size={16} className="text-blue-500" />
          </div>
          <div>
            <p className="text-xs text-gray-500">ADWIN Drift</p>
            <p className="text-sm font-semibold text-gray-200">0.002 Δ</p>
          </div>
        </div>
      </div>
    </div>
  );
}
