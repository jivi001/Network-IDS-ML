"use client";
import React, { useState, useEffect } from 'react';
import DashboardLayout from '../components/layout/DashboardLayout';
import SystemHealthWidget from '../components/dashboard/SystemHealthWidget';
import ThreatAlertsTable from '../components/dashboard/ThreatAlertsTable';
import TrafficChart from '../components/dashboard/TrafficChart';
import AnomalyTimeline from '../components/dashboard/AnomalyTimeline';
import AttackDistributionChart from '../components/dashboard/AttackDistributionChart';
import ModelExplainabilityPanel from '../components/dashboard/ModelExplainabilityPanel';

const initialData = Array.from({ length: 20 }).map((_, i) => ({
  time: new Date(Date.now() - (20 - i) * 2000).toLocaleTimeString(),
  volume: Math.floor(Math.random() * 2000) + 1000,
  anomalyScore: Math.random() * 0.1
}));

export default function SOCDashboard() {
  const [alerts, setAlerts] = useState<any[]>([]);
  const [trafficData, setTrafficData] = useState(initialData);
  const [status, setStatus] = useState("Connecting...");
  const [currentScore, setCurrentScore] = useState(0.01);
  const modelVersion = "NIDS-Core-v3.1";

  // WebSocket Integration
  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8000/ws/alerts");
    
    socket.onopen = () => setStatus("System Healthy");
    socket.onclose = () => setStatus("Disconnected");
    socket.onerror = () => setStatus("Error");

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const score = data.anomaly_score || 0;
      setCurrentScore(score);

      if (data.is_anomaly || score > 0.8) {
        setAlerts(prev => [{
          id: Date.now().toString(),
          type: data.attack_type || 'Unknown Anomaly',
          score: score.toFixed(2),
          time: new Date().toLocaleTimeString(),
          srcIp: `192.168.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
          destPort: [80, 443, 22, 4444, 3389][Math.floor(Math.random() * 5)].toString()
        }, ...prev].slice(0, 50));
      }
      
      setTrafficData(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        volume: data.volume || Math.floor(Math.random() * 4000) + 500,
        anomalyScore: score
      }].slice(-30)); // Keep last 30 points
    };

    return () => socket.close();
  }, []);

  return (
    <DashboardLayout status={status} version={modelVersion}>
      
      {/* Top Row: System Health, Explanations, Distribution */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4 h-auto md:h-64">
        <div className="col-span-1">
          <SystemHealthWidget />
        </div>
        <div className="col-span-1">
          <ModelExplainabilityPanel score={currentScore} />
        </div>
        <div className="col-span-1">
          <AttackDistributionChart />
        </div>
      </div>

      {/* Middle Row: Telemetry Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4 h-auto lg:h-[340px]">
        <div className="col-span-1 h-[300px] lg:h-full">
          <TrafficChart data={trafficData} />
        </div>
        <div className="col-span-1 h-[300px] lg:h-full">
          <AnomalyTimeline data={trafficData} />
        </div>
      </div>

      {/* Bottom Row: Threat Alerts Table */}
      <div className="h-[400px]">
        <ThreatAlertsTable alerts={alerts} />
      </div>

    </DashboardLayout>
  );
}

