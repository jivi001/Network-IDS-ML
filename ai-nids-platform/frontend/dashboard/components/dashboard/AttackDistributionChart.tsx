import React, { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { PieChart as PieChartIcon } from 'lucide-react';

const data = [
  { name: 'DoS', value: 45, color: '#ff4d4f' },
  { name: 'Port Scan', value: 30, color: '#ffb020' },
  { name: 'Brute Force', value: 15, color: '#a855f7' },
  { name: 'Botnet', value: 10, color: '#00e5ff' },
];

export default function AttackDistributionChart() {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  return (
    <div className="bg-[#111827]/80 backdrop-blur-xl border border-gray-800 rounded-2xl p-5 shadow-lg h-full flex flex-col">
      <div className="flex items-center gap-2 mb-2">
        <PieChartIcon className="text-gray-400" size={18} />
        <h2 className="text-gray-200 font-semibold tracking-wide">Attack Distribution</h2>
      </div>
      <p className="text-xs text-gray-500 mb-4">Historical analysis of flagged traffic types.</p>
      
      <div className="flex-1 w-full min-h-[200px]">
        {isClient ? (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={data}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                  stroke="none"
                >
                  {data.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0b1220', border: '1px solid #374151', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
                  itemStyle={{ color: '#fff' }}
                />
                <Legend verticalAlign="bottom" height={36} iconType="circle" wrapperStyle={{ fontSize: '12px', color: '#9ca3af' }} />
              </PieChart>
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
