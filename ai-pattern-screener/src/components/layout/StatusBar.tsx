"use client";

import { useStore } from "@/store/useStore";
import { Wifi, Cpu, Database, Globe, Activity, Zap, Radio, TrendingUp, TrendingDown } from "lucide-react";
import { useEffect, useState } from "react";
import LiveTicker from "@/components/market/LiveTicker";

export default function StatusBar() {
  const { scannerData, breadthData, scanning, liveScanning, fetchBreadth } = useStore();

  useEffect(() => {
    fetchBreadth();
    const interval = setInterval(fetchBreadth, 30000);
    return () => clearInterval(interval);
  }, [fetchBreadth]);

  return (
    <footer
      className="h-8 flex items-center justify-between px-4 border-t text-[11px] bg-[#0B0F17]"
      style={{ borderColor: "rgba(255,255,255,0.08)" }}
    >
      <div className="flex items-center gap-4 flex-1 min-w-0">
        <div className="flex items-center gap-1.5 shrink-0">
          <div className="status-dot live" />
          <span className="text-[#22C55E] font-medium">CONNECTED</span>
        </div>
        <span className="text-[#4B5563] shrink-0">|</span>
        <div className="flex items-center gap-1 text-[#94A3B8] shrink-0">
          <Globe size={10} />
          <span>NSE</span>
        </div>
        <div className="flex items-center gap-1 text-[#94A3B8] shrink-0">
          <Activity size={10} />
          <span>{breadthData ? `${scannerData.length || 493} stocks` : "—"}</span>
        </div>
        <div className="flex items-center gap-1 text-[#94A3B8] shrink-0">
          <Wifi size={10} className={liveScanning ? "text-[#8B5CF6] animate-pulse" : ""} />
          <span>{liveScanning ? "Live Scan" : scanning ? "Scanning" : "Paused"}</span>
        </div>
        <div className="flex-1 min-w-0 max-w-md">
          <LiveTicker />
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1 text-[#94A3B8]">
          <Zap size={10} className="text-[#3B82F6]" />
          <span>Scanning: {scannerData.filter((r) => r.alerted).length} / {scannerData.length || 493}</span>
        </div>
        <span className="text-[#4B5563]">|</span>
        <span className="text-[#94A3B8]">Next update: {scanning ? "30s" : "—"}</span>
        <span className="text-[#4B5563]">|</span>
        <span className="text-[#94A3B8]">Market: <span className="text-[#22C55E]">Open</span></span>
        <span className="text-[#4B5563]">|</span>
        <span className="text-[#94A3B8]" suppressHydrationWarning>{new Date().toLocaleString("en-IN", { timeZone: "Asia/Kolkata", day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" })} IST</span>
      </div>
    </footer>
  );
}
