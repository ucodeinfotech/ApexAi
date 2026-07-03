"use client";

import TopNav from "@/components/layout/TopNav";
import Sidebar from "@/components/layout/Sidebar";
import StatusBar from "@/components/layout/StatusBar";
import AIDock from "@/components/layout/AIDock";
import Workspace from "@/components/layout/Workspace";
import CandlestickChart from "@/components/charts/CandlestickChart";
import { useStore } from "@/store/useStore";
import { useEffect } from "react";

function Toast() {
  const { toast } = useStore();
  useEffect(() => {}, [toast.visible]);

  if (!toast.visible) return null;

  return (
    <div className="fixed top-20 left-1/2 -translate-x-1/2 z-[100] pointer-events-none transition-all duration-300">
      <div className="px-4 py-2 rounded-lg bg-[#3B82F6]/90 backdrop-blur-md text-white text-xs font-medium shadow-lg shadow-[#3B82F6]/20 whitespace-nowrap">
        {toast.message}
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <div className="h-screen w-screen flex flex-col bg-[#0B0F17] overflow-hidden">
      <TopNav />
      <div className="flex flex-1 min-h-0">
        <Sidebar />
        <main className="flex-1 flex flex-col min-w-0 overflow-y-auto">
          <Workspace />
        </main>
      </div>
      <StatusBar />
      <AIDock />
      <CandlestickChart />
      <Toast />
    </div>
  );
}
