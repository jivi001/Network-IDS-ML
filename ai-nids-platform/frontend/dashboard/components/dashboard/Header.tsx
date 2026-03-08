import React from 'react';
import { ShieldCheck, Activity, Server, Radio } from 'lucide-react';

interface HeaderProps {
  status: string;
  version: string;
}

export default function Header({ status, version }: HeaderProps) {
  const isHealthy = status === "System Healthy";

  return (
    <header className="flex flex-col md:flex-row items-start md:items-center justify-between mb-10 gap-6 relative z-10">
      <div className="flex items-center gap-4">
        <div className={`p-3 rounded-2xl bg-gradient-to-br ${isHealthy ? 'from-emerald-500/20 to-emerald-500/5 border-emerald-500/30' : 'from-red-500/20 to-red-500/5 border-red-500/30'} border backdrop-blur-xl`}>
          <ShieldCheck className={isHealthy ? "text-emerald-400" : "text-red-400"} size={32} />
        </div>
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white mb-1">AI-NIDS Command Center</h1>
          <p className="text-gray-400 flex items-center gap-2 text-sm">
            <Radio size={14} className="text-blue-400 animate-pulse" />
            Live Monitoring & Threat Detection
          </p>
        </div>
      </div>
      
      <div className="flex gap-4 text-sm font-medium">
        <div className="bg-gray-800/40 backdrop-blur-xl border border-gray-700/50 px-5 py-2.5 rounded-full flex items-center gap-2 shadow-lg">
          <Server size={16} className="text-gray-400" />
          Model: <span className="text-gray-200 font-mono tracking-tight cursor-default hover:text-blue-400 transition-colors">{version}</span>
        </div>
        <div className={`backdrop-blur-xl border px-5 py-2.5 rounded-full flex items-center gap-2 shadow-lg transition-colors ${
          isHealthy 
            ? "bg-emerald-900/20 border-emerald-800/50 text-emerald-400"
            : "bg-red-900/20 border-red-800/50 text-red-400"
        }`}>
          <Activity size={16} className={isHealthy ? "animate-pulse" : ""} /> {status}
        </div>
      </div>
    </header>
  );
}
