"use client";

import { useStore } from "@/store/useStore";
import { TrendingUp, TrendingDown, ArrowUp, ArrowDown, BarChart3 } from "lucide-react";

export default function BreadthCards() {
  const { breadthData, fetchBreadth } = useStore();
  const data = breadthData;

  return (
    <div className="h-full panel flex flex-col">
      <div className="panel-header shrink-0 drag-handle cursor-grab active:cursor-grabbing">
        <span className="text-xs font-semibold">MARKET BREADTH</span>
        <button onClick={() => fetchBreadth()} className="text-[9px] text-[#3B82F6] hover:text-[#2563EB] btn-ghost px-1.5 py-0.5">Refresh</button>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
        {!data ? (
          <div className="flex items-center justify-center h-full text-[#4B5563] text-xs">Loading...</div>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-1.5">
              <div className="glass rounded-lg px-2.5 py-2 text-center">
                <div className="text-[9px] text-[#4B5563]">Adv/Dec</div>
                <div className="flex items-center justify-center gap-1 mt-0.5">
                  <span className="text-xs font-semibold text-[#22C55E]">{data.advancers}</span>
                  <span className="text-[9px] text-[#4B5563]">/</span>
                  <span className="text-xs font-semibold text-[#EF4444]">{data.decliners}</span>
                </div>
              </div>
              <div className="glass rounded-lg px-2.5 py-2 text-center">
                <div className="text-[9px] text-[#4B5563]">New High/Low</div>
                <div className="flex items-center justify-center gap-1 mt-0.5">
                  <span className="text-xs font-semibold text-[#22C55E]">{data.newHighs}</span>
                  <span className="text-[9px] text-[#4B5563]">/</span>
                  <span className="text-xs font-semibold text-[#EF4444]">{data.newLows}</span>
                </div>
              </div>
              <div className="glass rounded-lg px-2.5 py-2 text-center">
                <div className="text-[9px] text-[#4B5563]">VIX</div>
                <div className="text-xs font-semibold text-[#F59E0B] mt-0.5">{data.vix}</div>
              </div>
            </div>

            <div>
              <div className="text-[9px] text-[#4B5563] font-medium mb-1 uppercase tracking-wider flex items-center gap-1">
                <TrendingUp size={10} className="text-[#22C55E]" /> Top Gainers
              </div>
              {data.topGainers.slice(0, 5).map((g) => (
                <div key={g.ticker} className="flex items-center justify-between glass-hover rounded-lg px-2.5 py-1.5 text-xs">
                  <div className="flex items-center gap-2">
                    <ArrowUp size={10} className="text-[#22C55E]" />
                    <span className="font-medium">{g.ticker}</span>
                  </div>
                  <span className="text-[#22C55E] font-medium">+{g.change.toFixed(1)}%</span>
                </div>
              ))}
            </div>

            <div>
              <div className="text-[9px] text-[#4B5563] font-medium mb-1 uppercase tracking-wider flex items-center gap-1">
                <TrendingDown size={10} className="text-[#EF4444]" /> Top Losers
              </div>
              {data.topLosers.slice(0, 5).map((g) => (
                <div key={g.ticker} className="flex items-center justify-between glass-hover rounded-lg px-2.5 py-1.5 text-xs">
                  <div className="flex items-center gap-2">
                    <ArrowDown size={10} className="text-[#EF4444]" />
                    <span className="font-medium">{g.ticker}</span>
                  </div>
                  <span className="text-[#EF4444] font-medium">{g.change.toFixed(1)}%</span>
                </div>
              ))}
            </div>

            <div>
              <div className="text-[9px] text-[#4B5563] font-medium mb-1 uppercase tracking-wider flex items-center gap-1">
                <BarChart3 size={10} /> Most Active
              </div>
              {data.mostActive.slice(0, 5).map((m) => (
                <div key={m.ticker} className="flex items-center justify-between glass-hover rounded-lg px-2.5 py-1.5 text-xs">
                  <span className="font-medium">{m.ticker}</span>
                  <span className="text-[#94A3B8]">{(m.volume / 1e6).toFixed(1)}M</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
