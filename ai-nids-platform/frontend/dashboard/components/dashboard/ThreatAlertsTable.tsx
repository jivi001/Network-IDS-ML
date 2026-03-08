import React from 'react';
import { ShieldAlert, Crosshair, ChevronRight } from 'lucide-react';

interface Alert {
  id: string;
  type: string;
  score: string;
  time: string;
  srcIp: string;
  destPort: string;
}

interface ThreatAlertsTableProps {
  alerts: Alert[];
}

export default function ThreatAlertsTable({ alerts }: ThreatAlertsTableProps) {
  const getSeverityStyle = (score: string) => {
    const val = parseFloat(score);
    if (val > 0.8) return { text: "text-[#ff4d4f]", bg: "bg-[#ff4d4f]/10", border: "border-[#ff4d4f]/30", label: "CRITICAL" };
    if (val > 0.6) return { text: "text-orange-500", bg: "bg-orange-500/10", border: "border-orange-500/30", label: "HIGH" };
    if (val > 0.4) return { text: "text-yellow-500", bg: "bg-yellow-500/10", border: "border-yellow-500/30", label: "MEDIUM" };
    return { text: "text-[#00e5ff]", bg: "bg-[#00e5ff]/10", border: "border-[#00e5ff]/30", label: "LOW" };
  };

  return (
    <div className="bg-[#111827]/80 backdrop-blur-xl border border-gray-800 rounded-2xl flex flex-col h-full shadow-lg overflow-hidden relative">
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-red-600 via-orange-500 to-transparent"></div>
      
      <div className="p-5 border-b border-gray-800 flex justify-between items-center bg-[#111827]">
        <h2 className="text-gray-200 font-semibold tracking-wide flex items-center gap-2">
          <ShieldAlert className="text-[#ff4d4f]" size={18} /> Active Threat Intelligence
        </h2>
        <span className="text-xs bg-gray-800 text-gray-400 px-3 py-1 rounded-full font-mono border border-gray-700">
          Showing {alerts.length} threats
        </span>
      </div>

      <div className="flex-1 overflow-auto custom-scrollbar">
        <table className="w-full text-left text-sm text-gray-300 border-collapse">
          <thead className="bg-[#111827] sticky top-0 z-10 shadow-sm border-b border-gray-800">
            <tr>
              <th className="px-5 py-3 font-medium text-gray-500 uppercase text-xs tracking-wider">Severity</th>
              <th className="px-5 py-3 font-medium text-gray-500 uppercase text-xs tracking-wider">Attack Type</th>
              <th className="px-5 py-3 font-medium text-gray-500 uppercase text-xs tracking-wider">Source IP</th>
              <th className="px-5 py-3 font-medium text-gray-500 uppercase text-xs tracking-wider">D.Port</th>
              <th className="px-5 py-3 font-medium text-gray-500 uppercase text-xs tracking-wider">Timestamp</th>
              <th className="px-5 py-3 font-medium text-gray-500 uppercase text-xs tracking-wider text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/60">
            {alerts.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-5 py-16 text-center text-gray-500">
                  <ShieldAlert className="mx-auto mb-3 opacity-20" size={48} />
                  <p>No active threats detected in the current stream.</p>
                </td>
              </tr>
            ) : (
                alerts.map((alert, i) => {
                  const severity = getSeverityStyle(alert.score);
                  return (
                    <tr key={i} className="hover:bg-gray-800/40 transition-colors group cursor-pointer">
                      <td className="px-5 py-3 whitespace-nowrap">
                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-bold ${severity.text} ${severity.bg} ${severity.border} border`}>
                          <Crosshair size={10} /> {severity.label}
                        </span>
                      </td>
                      <td className="px-5 py-3 font-medium text-gray-200">
                        {alert.type.replace(/_/g, ' ')}
                      </td>
                      <td className="px-5 py-3 font-mono text-[11px] text-gray-400">
                        {alert.srcIp}
                      </td>
                      <td className="px-5 py-3 font-mono text-[11px] text-gray-400">
                        {alert.destPort}
                      </td>
                      <td className="px-5 py-3 text-[11px] text-gray-500">
                        {alert.time}
                      </td>
                      <td className="px-5 py-3 text-right">
                        <button className="p-1 rounded hover:bg-gray-700 text-gray-500 hover:text-white transition-colors">
                          <ChevronRight size={16} />
                        </button>
                      </td>
                    </tr>
                  )
                })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
