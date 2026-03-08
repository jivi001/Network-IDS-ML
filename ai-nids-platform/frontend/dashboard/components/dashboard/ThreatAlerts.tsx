import React from 'react';
import { ShieldAlert, ShieldCheck } from 'lucide-react';

interface ThreatAlertsProps {
  alerts: any[];
}

export default function ThreatAlerts({ alerts }: ThreatAlertsProps) {
  return (
    <div className="bg-gray-800/30 backdrop-blur-xl border border-gray-700/50 rounded-3xl p-6 shadow-2xl relative overflow-hidden group h-full">
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-red-500/80 to-orange-500/80 opacity-70" />
      
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold flex items-center gap-2 text-white drop-shadow-md">
          <ShieldAlert className="text-red-400" size={22} />
          Live Threats
        </h2>
        {alerts.length > 0 && (
          <div className="bg-red-500/10 text-red-400 px-3 py-1 rounded-full text-xs font-bold border border-red-500/20 shadow-[0_0_15px_rgba(239,68,68,0.15)]">
            {alerts.length} Detected
          </div>
        )}
      </div>

      <div className="space-y-3 h-[420px] overflow-y-auto pr-2 custom-scrollbar">
        {alerts.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-gray-500 pb-10">
            <div className="p-4 rounded-full bg-emerald-500/5 mb-4">
              <ShieldCheck size={48} className="opacity-40 text-emerald-500" />
            </div>
            <p className="font-medium text-gray-300">No active threats detected.</p>
            <p className="text-sm mt-1 opacity-60">Network traffic is currently benign.</p>
          </div>
        ) : (
          alerts.map((alert, index) => (
            <div 
              key={alert.id} 
              className={`relative bg-gray-900/40 p-4 rounded-2xl border border-red-900/30 overflow-hidden transform transition-all duration-300 hover:bg-gray-800/60 hover:border-red-500/30 hover:shadow-lg ${index === 0 ? 'animate-slide-in' : ''}`}
            >
              <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-red-500 to-orange-500" />
              <div className="flex justify-between items-start mb-2">
                <span className="font-bold text-red-400 text-sm tracking-wide bg-red-500/10 border border-red-500/20 px-2 py-0.5 rounded shadow-sm">
                  {alert.type.replace(/_/g, ' ')}
                </span>
                <span className="text-xs text-gray-500 font-mono bg-gray-800/50 px-2 py-0.5 rounded">
                  {alert.time}
                </span>
              </div>
              <div className="flex justify-between items-end mt-3">
                <div className="text-xs text-gray-400 font-medium">
                  Confidence Score
                </div>
                <div className="font-mono text-lg font-semibold text-white drop-shadow-[0_0_8px_rgba(239,68,68,0.5)]">
                  {alert.score}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
