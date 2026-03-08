import React, { useState, useEffect } from 'react';
import { 
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  AreaChart, Area 
} from 'recharts';
import { Activity } from 'lucide-react';

interface TrafficChartProps {
  data: any[];
}

export default function TrafficChart({ data }: TrafficChartProps) {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  return (
    <div className="bg-[#111827]/80 backdrop-blur-xl border border-gray-800 rounded-2xl p-5 shadow-lg h-full relative overflow-hidden flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-gray-200 font-semibold tracking-wide flex items-center gap-2">
            <Activity className="text-[#00e5ff]" size={18} /> Network Traffic Volume
          </h2>
          <p className="text-xs text-gray-500 mt-1">Packets per interval (pps)</p>
        </div>
        <div className="flex items-center gap-4 text-xs font-medium">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.6)]" />
            <span className="text-gray-400">Total Volume</span>
          </div>
        </div>
      </div>
      
      <div className="flex-1 w-full mt-2 min-h-[200px]">
        {isClient ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="colorVolumeGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00e5ff" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#00e5ff" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(55, 65, 81, 0.3)" vertical={false} />
              <XAxis 
                dataKey="time" 
                stroke="#4b5563" 
                fontSize={10} 
                tickMargin={10} 
                axisLine={false} 
                tickLine={false} 
              />
              <YAxis 
                stroke="#4b5563" 
                fontSize={10} 
                tickMargin={10} 
                axisLine={false} 
                tickLine={false} 
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#0b1220', 
                  border: '1px solid rgba(75, 85, 99, 0.4)', 
                  borderRadius: '8px',
                  color: '#fff',
                  fontSize: '12px'
                }} 
                cursor={{ stroke: 'rgba(0, 229, 255, 0.1)', strokeWidth: 2 }}
              />
              <Area 
                type="monotone" 
                dataKey="volume" 
                stroke="#00e5ff" 
                strokeWidth={2} 
                fillOpacity={1} 
                fill="url(#colorVolumeGrad)" 
                isAnimationActive={false}
                activeDot={{ r: 4, strokeWidth: 0, fill: '#00e5ff' }}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <span className="text-gray-600 text-sm">Loading chart...</span>
          </div>
        )}
      </div>
    </div>
  );
}
