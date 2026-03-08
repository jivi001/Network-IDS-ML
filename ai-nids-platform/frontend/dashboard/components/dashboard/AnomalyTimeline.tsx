import React, { useState, useEffect } from 'react';
import { 
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  AreaChart, Area, ReferenceLine
} from 'recharts';
import { Target } from 'lucide-react';

interface AnomalyTimelineProps {
  data: any[];
}

export default function AnomalyTimeline({ data }: AnomalyTimelineProps) {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  return (
    <div className="bg-[#111827]/80 backdrop-blur-xl border border-gray-800 rounded-2xl p-5 shadow-lg h-full relative overflow-hidden flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-gray-200 font-semibold tracking-wide flex items-center gap-2">
            <Target className="text-[#ff4d4f]" size={18} /> Anomaly Score Timeline
          </h2>
          <p className="text-xs text-gray-500 mt-1">Live model prediction confidence</p>
        </div>
        <div className="flex items-center gap-4 text-xs font-medium">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-[2px] bg-[#ff4d4f]" />
            <span className="text-gray-400">Threshold (0.8)</span>
          </div>
        </div>
      </div>
      
      <div className="flex-1 w-full mt-2 min-h-[200px]">
        {isClient ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="colorAnomalyGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ff4d4f" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#ff4d4f" stopOpacity={0}/>
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
                domain={[0, 1]}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#0b1220', 
                  border: '1px solid rgba(75, 85, 99, 0.4)', 
                  borderRadius: '8px',
                  color: '#fff',
                  fontSize: '12px'
                }} 
                cursor={{ stroke: 'rgba(255, 77, 79, 0.1)', strokeWidth: 2 }}
              />
              <ReferenceLine y={0.8} stroke="#ff4d4f" strokeDasharray="3 3" strokeOpacity={0.5} />
              <Area 
                type="monotone" 
                dataKey="anomalyScore" 
                stroke="#ffb020" 
                strokeWidth={2} 
                fillOpacity={1} 
                fill="url(#colorAnomalyGrad)" 
                isAnimationActive={false}
                activeDot={{ r: 4, strokeWidth: 0, fill: '#ffb020' }}
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
