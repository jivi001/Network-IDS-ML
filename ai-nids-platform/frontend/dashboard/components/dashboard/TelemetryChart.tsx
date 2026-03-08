import React, { useState, useEffect } from 'react';
import { 
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  AreaChart, Area 
} from 'recharts';
import { Activity } from 'lucide-react';

interface TelemetryChartProps {
  data: any[];
}

export default function TelemetryChart({ data }: TelemetryChartProps) {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  return (
    <div className="bg-gray-800/30 backdrop-blur-xl border border-gray-700/50 rounded-3xl p-6 lg:p-8 shadow-2xl h-full relative overflow-hidden">
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500/80 to-emerald-500/80 opacity-70" />
      
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1 drop-shadow-md">Network Telemetry</h2>
          <p className="text-sm text-gray-400">Volume and automated anomaly scoring over time</p>
        </div>
        <div className="flex items-center gap-4 text-xs font-semibold bg-gray-900/50 px-4 py-2 rounded-full border border-gray-800/50">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.8)]" />
            <span className="text-gray-300">Volume</span>
          </div>
          <div className="flex items-center gap-2 ml-2">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.8)]" />
            <span className="text-gray-300">Anomaly Level</span>
          </div>
        </div>
      </div>
      
      <div className="h-[400px] w-full mt-4">
        {isClient ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="colorVolume" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorAnomaly" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(55, 65, 81, 0.4)" vertical={false} />
              <XAxis 
                dataKey="time" 
                stroke="#6b7280" 
                fontSize={12} 
                tickMargin={12} 
                axisLine={false} 
                tickLine={false} 
              />
              <YAxis 
                yAxisId="left" 
                stroke="#6b7280" 
                fontSize={12} 
                tickMargin={12} 
                axisLine={false} 
                tickLine={false} 
                domain={[0, 'auto']}
              />
              <YAxis 
                yAxisId="right" 
                orientation="right" 
                stroke="#6b7280" 
                fontSize={12} 
                axisLine={false} 
                tickLine={false} 
                domain={[0, 1]} 
                hide 
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'rgba(17, 24, 39, 0.85)', 
                  backdropFilter: 'blur(12px)',
                  border: '1px solid rgba(75, 85, 99, 0.4)', 
                  borderRadius: '16px',
                  color: '#fff',
                  boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5)'
                }} 
                itemStyle={{ fontSize: '13px', fontWeight: 600, padding: '2px 0' }}
                cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 2 }}
              />
              <Area 
                yAxisId="left"
                type="monotone" 
                dataKey="volume" 
                stroke="#3b82f6" 
                strokeWidth={3} 
                fillOpacity={1} 
                fill="url(#colorVolume)" 
                name="Flow Volume" 
                isAnimationActive={true}
                animationDuration={1000}
                activeDot={{ r: 6, strokeWidth: 0, fill: '#3b82f6', style: { filter: 'drop-shadow(0 0 8px rgba(59,130,246,0.8))' } }}
              />
              <Area 
                yAxisId="right"
                type="monotone" 
                dataKey="anomalyScore" 
                stroke="#ef4444" 
                strokeWidth={2} 
                fillOpacity={1} 
                fill="url(#colorAnomaly)" 
                name="Anomaly Score" 
                isAnimationActive={true}
                animationDuration={1000}
                activeDot={{ r: 6, strokeWidth: 0, fill: '#ef4444', style: { filter: 'drop-shadow(0 0 8px rgba(239,68,68,0.8))' } }}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center text-gray-500">
            <Activity size={32} className="animate-pulse mb-3 opacity-50 text-blue-500" />
            <p className="text-sm font-medium animate-pulse opacity-70">Initializing Telemetry Engine...</p>
          </div>
        )}
      </div>
    </div>
  );
}
