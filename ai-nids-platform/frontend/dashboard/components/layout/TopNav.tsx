import React from 'react';
import { Search, Bell, User, Clock, ShieldCheck, Activity } from 'lucide-react';

interface TopNavProps {
  status: string;
  version: string;
  environment?: string;
}

export default function TopNav({ status, version, environment = "PRODUCTION" }: TopNavProps) {
  const isHealthy = status === "System Healthy";

  return (
    <header className="absolute top-0 left-0 right-0 h-16 bg-[#111827]/80 backdrop-blur-xl border-b border-gray-800/60 z-20 flex items-center justify-between px-6 lg:pl-72 shadow-sm">
      <div className="flex items-center gap-4">
        {/* Status Indicators */}
        <div className={`hidden md:flex px-3 py-1.5 rounded bg-gray-900 border border-gray-800 items-center gap-2 text-xs font-mono`}>
           <span className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}></span>
           <span className="text-gray-300">{status}</span>
        </div>
        
        <div className="hidden md:flex px-3 py-1.5 rounded bg-gray-900 border border-gray-800 items-center gap-2 text-xs font-mono text-gray-400">
           <span>ENV: <span className="text-[#00e5ff] font-semibold">{environment}</span></span>
        </div>

        <div className="hidden lg:flex px-3 py-1.5 rounded bg-gray-900 border border-gray-800 items-center gap-2 text-xs font-mono text-gray-400">
           <span>Model: <span className="text-gray-300">{version}</span></span>
        </div>
      </div>

      <div className="flex items-center gap-5">
        {/* Search Bar Placeholder */}
        <div className="relative hidden sm:block">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={14} />
          <input 
            type="text" 
            placeholder="Search IPs, Alerts, Hashes..." 
            className="w-64 bg-gray-900/50 border border-gray-700 text-sm rounded-lg pl-9 pr-3 py-1.5 text-gray-200 focus:outline-none focus:border-[#00e5ff] focus:ring-1 focus:ring-[#00e5ff] transition-all placeholder-gray-600"
          />
        </div>

        <div className="h-6 w-px bg-gray-700/50 hidden sm:block"></div>

        {/* Action Icons */}
        <button className="relative text-gray-400 hover:text-white transition-colors">
          <Bell size={18} />
          <span className="absolute -top-1 -right-1 flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#ff4d4f] opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-[#ff4d4f] border border-[#111827]"></span>
          </span>
        </button>

        <button className="flex items-center justify-center w-8 h-8 rounded-full bg-gradient-to-r from-blue-600 to-[#00e5ff] text-white shadow-[0_0_10px_rgba(0,229,255,0.3)]">
          <User size={16} />
        </button>
      </div>
    </header>
  );
}
