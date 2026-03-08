import React from 'react';
import { BrainCircuit, Fingerprint, Zap } from 'lucide-react';

interface FeatureImportance {
  feature: string;
  weight: number;
}

interface ModelExplainabilityPanelProps {
  features?: FeatureImportance[];
  score?: number;
}

export default function ModelExplainabilityPanel({ 
  features = [
    { feature: "Flow Duration < 5ms", weight: 0.85 },
    { feature: "Dest Port = 4444", weight: 0.72 },
    { feature: "Packet Len Variance", weight: 0.64 },
    { feature: "Forward Packets/s", weight: 0.45 },
  ],
  score = 0.92
}: ModelExplainabilityPanelProps) {
  
  return (
    <div className="bg-[#111827]/80 backdrop-blur-xl border border-gray-800 rounded-2xl p-5 shadow-lg flex flex-col h-full relative">
      <div className="flex items-center gap-3 mb-5">
        <div className="p-2 bg-purple-500/10 rounded-xl border border-purple-500/20">
          <BrainCircuit className="text-purple-400" size={18} />
        </div>
        <h2 className="text-gray-200 font-semibold tracking-wide">Model Explainability</h2>
      </div>

      <div className="flex-1">
        <div className="mb-6 flex items-center justify-between bg-gray-900/60 p-4 rounded-xl border border-gray-800">
          <div>
            <p className="text-xs text-gray-500 mb-1 flex items-center gap-1"><Zap size={12} className="text-[#00e5ff]" /> Current Anomaly Score</p>
            <div className="text-3xl font-mono font-bold text-white tracking-tighter">
              {score.toFixed(2)}
            </div>
          </div>
          <div className="text-right">
             <span className="inline-block px-3 py-1 rounded bg-[#ff4d4f]/20 text-[#ff4d4f] border border-[#ff4d4f]/30 text-xs font-bold tracking-wide">ISOLATION FOREST FLAG</span>
          </div>
        </div>

        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4 flex items-center gap-2">
          <Fingerprint size={14} /> Top Contributing Features (SHAP)
        </h3>

        <div className="space-y-4">
          {features.map((f, i) => (
            <div key={i} className="relative">
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-gray-300 font-mono tracking-tight">{f.feature}</span>
                <span className="text-gray-500">{f.weight.toFixed(2)}</span>
              </div>
              <div className="w-full bg-gray-800/80 rounded-full h-1.5 overflow-hidden">
                <div 
                  className={`h-full rounded-full ${i === 0 ? 'bg-[#ff4d4f]' : i === 1 ? 'bg-orange-500' : 'bg-purple-500'}`} 
                  style={{ width: `${f.weight * 100}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
