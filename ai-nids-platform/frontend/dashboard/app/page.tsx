"use client";
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { AlertTriangle, ShieldCheck, Activity } from 'lucide-react';

export default function SOCDashboard() {
  const [alerts, setAlerts] = useState([]);
  const [trafficData, setTrafficData] = useState([]);

  // Real WebSocket connection for live Streaming Ingestion
  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8000/ws/alerts");
    
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.is_anomaly) {
        setAlerts(prev => [{
          id: Date.now(),
          type: data.attack_type,
          score: data.anomaly_score.toFixed(2),
          time: new Date().toLocaleTimeString()
        }, ...prev].slice(0, 10));
      }
      
      setTrafficData(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        volume: data.volume || Math.floor(Math.random() * 5000), // mock volume if missing
        anomalyScore: data.anomaly_score || 0
      }].slice(-20));
    };

    return () => socket.close();
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8 font-sans">
      <header className="flex items-center justify-between mb-8 border-b border-gray-700 pb-4">
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <ShieldCheck className="text-green-400" size={36} />
          AI-NIDS Security Operations Center
        </h1>
        <div className="flex gap-4">
          <span className="bg-blue-900/50 text-blue-400 px-4 py-2 rounded-full border border-blue-800">
            Model: v2.0.0-Hybrid
          </span>
          <span className="bg-green-900/50 text-green-400 px-4 py-2 rounded-full border border-green-800 flex items-center gap-2">
            <Activity size={18} /> System Healthy
          </span>
        </div>
      </header>

      <div className="grid grid-cols-3 gap-6">
        {/* Alerts Panel */}
        <div className="col-span-1 bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-xl overflow-hidden">
          <h2 className="text-xl font-semibold mb-4 text-red-400 flex items-center gap-2">
            <AlertTriangle /> Live Threat Alerts
          </h2>
          <div className="space-y-3 h-96 overflow-y-auto pr-2">
            {alerts.length === 0 ? <p className="text-gray-400 italic">No threats detected.</p> : alerts.map(alert => (
              <div key={alert.id} className="bg-red-900/20 border border-red-800 p-3 rounded flex justify-between items-center">
                <div>
                  <p className="font-bold text-red-300">{alert.type}</p>
                  <p className="text-xs text-gray-400">{alert.time}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm">Severity: <span className="font-mono text-red-400">{alert.score}</span></p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Telemetry Charts */}
        <div className="col-span-2 bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-xl">
          <h2 className="text-xl font-semibold mb-4">Network Traffic Volume & Anomaly Metrics</h2>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trafficData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip contentStyle={{backgroundColor: '#1F2937', border: 'none', color: '#fff'}} />
                <Line type="monotone" dataKey="volume" stroke="#3B82F6" strokeWidth={3} dot={false} name="Packets/sec" />
                <Line type="monotone" dataKey="anomalyScore" stroke="#EF4444" strokeWidth={2} dot={false} name="Anomaly Score" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}