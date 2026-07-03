"use client";

import { useEffect, useState, useRef } from "react";
import { TrendingUp, TrendingDown, Activity, BarChart3 } from "lucide-react";

interface TickData {
  active: boolean;
  tickCount: number;
  uniqueStocks: number;
  totalVolume: number;
  timestamp: string;
  topGainers: { symbol: string; ltp: number; changePct: number; volume: number }[];
  topLosers: { symbol: string; ltp: number; changePct: number; volume: number }[];
  summary: { advancers: number; decliners: number; unchanged: number; totalVolume: number };
}

export default function LiveTicker() {
  const [data, setData] = useState<TickData | null>(null);
  const [connected, setConnected] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const fetchTicks = async () => {
      try {
        const res = await fetch("/api/live/ticks");
        if (!res.ok) return;
        const d: TickData = await res.json();
        setData(d);
        setConnected(d.active);
      } catch {
        setConnected(false);
      }
    };

    fetchTicks();
    intervalRef.current = setInterval(fetchTicks, 3000);

    return () => { if (intervalRef.current != null) clearInterval(intervalRef.current); };
  }, []);

  return (
    <div className="glass rounded-lg overflow-hidden" style={{ border: "1px solid rgba(255,255,255,0.06)" }}>
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-white/5">
        <div className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-[#22C55E] animate-pulse" : "bg-[#EF4444]"}`} />
        <span className="text-[10px] font-medium text-[#94A3B8] uppercase tracking-wider">Live Feed</span>
        {data && <span className="text-[9px] text-[#4B5563]">{data.uniqueStocks ?? 0} stocks · {((data.totalVolume ?? 0) / 1e6).toFixed(1)}M vol</span>}
      </div>
      <div className="flex divide-x divide-white/5 text-[10px]">
        <div className="flex-1 px-2.5 py-1.5">
          <div className="text-[8px] text-[#4B5563] uppercase mb-0.5 flex items-center gap-1">
            <TrendingUp size={8} className="text-[#22C55E]" /> Gainers
          </div>
          {(data?.topGainers || []).slice(0, 3).map((g) => (
            <div key={g.symbol} className="flex justify-between py-0.5">
              <span className="text-white truncate max-w-[60px]">{g.symbol}</span>
              <span className="text-[#22C55E] tabular-nums">+{g.changePct.toFixed(1)}%</span>
            </div>
          ))}
          {(!data || !data.topGainers || data.topGainers.length === 0) && (
            <div className="text-[#4B5563] py-1">Waiting for data...</div>
          )}
        </div>
        <div className="flex-1 px-2.5 py-1.5">
          <div className="text-[8px] text-[#4B5563] uppercase mb-0.5 flex items-center gap-1">
            <TrendingDown size={8} className="text-[#EF4444]" /> Losers
          </div>
          {(data?.topLosers || []).slice(0, 3).map((g) => (
            <div key={g.symbol} className="flex justify-between py-0.5">
              <span className="text-white truncate max-w-[60px]">{g.symbol}</span>
              <span className="text-[#EF4444] tabular-nums">{g.changePct.toFixed(1)}%</span>
            </div>
          ))}
          {(!data || !data.topLosers || data.topLosers.length === 0) && (
            <div className="text-[#4B5563] py-1">Waiting for data...</div>
          )}
        </div>
        <div className="px-2.5 py-1.5 flex items-center gap-2">
          {data?.summary && (
            <div className="text-center">
              <div className="text-[8px] text-[#4B5563] uppercase">A/D</div>
              <div className="flex items-center gap-1 text-xs">
                <span className="text-[#22C55E] font-medium">{data.summary.advancers ?? 0}</span>
                <span className="text-[#4B5563]">/</span>
                <span className="text-[#EF4444] font-medium">{data.summary.decliners ?? 0}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
