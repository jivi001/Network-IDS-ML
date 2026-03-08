import React from 'react';
import Sidebar from './Sidebar';
import TopNav from './TopNav';

interface DashboardLayoutProps {
  children: React.ReactNode;
  status?: string;
  version?: string;
}

export default function DashboardLayout({ 
  children, 
  status = "System Healthy", 
  version = "v2.0.0-Hybrid" 
}: DashboardLayoutProps) {
  return (
    <div className="flex h-screen overflow-hidden bg-[#0b1220] text-gray-100 font-sans selection:bg-[#00e5ff]/30">
      <Sidebar />
      <div className="flex-1 flex flex-col h-full relative">
        <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
          <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-[#00e5ff]/10 blur-[120px] rounded-full mix-blend-screen" />
          <div className="absolute bottom-[-10%] right-[-5%] w-[40%] h-[40%] bg-blue-600/10 blur-[120px] rounded-full mix-blend-screen" />
        </div>
        
        <TopNav status={status} version={version} />
        
        <main className="flex-1 overflow-y-auto p-4 lg:p-6 lg:ml-64 mt-16 relative z-10 custom-scrollbar pb-12">
          <div className="max-w-[1800px] mx-auto h-full">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}

