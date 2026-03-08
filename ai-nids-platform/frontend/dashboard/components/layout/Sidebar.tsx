import React from 'react';
import { 
  LayoutDashboard, ShieldAlert, Activity, Network, 
  Search, BarChart4, Settings, HelpCircle
} from 'lucide-react';
import Link from 'next/link';

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 bottom-0 w-64 bg-[#111827]/95 backdrop-blur-xl border-r border-gray-800/60 z-30 flex flex-col pt-6 hidden lg:flex shadow-2xl">
      <div className="px-6 mb-10 flex items-center gap-3">
        <div className="p-2 rounded-xl bg-gradient-to-br from-[#00e5ff]/20 to-blue-600/10 border border-[#00e5ff]/30 shadow-[0_0_15px_rgba(0,229,255,0.2)]">
          <Network className="text-[#00e5ff]" size={24} />
        </div>
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white leading-tight">AI-NIDS</h1>
          <p className="text-[10px] uppercase tracking-widest text-[#00e5ff] font-semibold">SOC Platform</p>
        </div>
      </div>

      <div className="px-4 flex-1 space-y-1">
        <p className="px-3 text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 mt-4">Monitoring</p>
        
        <Link href="#" className="flex items-center gap-3 px-3 py-2.5 rounded-lg bg-[#00e5ff]/10 text-[#00e5ff] border border-[#00e5ff]/20 hover:bg-[#00e5ff]/20 transition-all font-medium">
          <LayoutDashboard size={18} /> Dashboard
        </Link>
        
        <Link href="#" className="flex items-center justify-between px-3 py-2.5 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-800/50 transition-all font-medium group">
          <div className="flex items-center gap-3">
            <ShieldAlert size={18} className="group-hover:text-[#ff4d4f] transition-colors" /> Live Threats
          </div>
          <span className="bg-[#ff4d4f]/20 text-[#ff4d4f] text-[10px] px-2 py-0.5 rounded-full border border-[#ff4d4f]/30">4</span>
        </Link>
        
        <Link href="#" className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-800/50 transition-all font-medium">
          <Activity size={18} /> Network Traffic
        </Link>
        
        <p className="px-3 text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 mt-8">Intelligence</p>
        
        <Link href="#" className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-800/50 transition-all font-medium">
          <Search size={18} /> Investigation
        </Link>
        
        <Link href="#" className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-800/50 transition-all font-medium">
          <BarChart4 size={18} /> Model Insights
        </Link>
      </div>

      <div className="p-4 border-t border-gray-800/60 space-y-1">
        <Link href="#" className="flex items-center gap-3 px-3 py-2 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-800/50 transition-all text-sm">
          <Settings size={16} /> Configurations
        </Link>
        <Link href="#" className="flex items-center gap-3 px-3 py-2 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-800/50 transition-all text-sm">
          <HelpCircle size={16} /> Support
        </Link>
      </div>
    </aside>
  );
}
