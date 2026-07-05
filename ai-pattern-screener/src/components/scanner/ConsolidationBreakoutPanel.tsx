"use client";

import { useEffect, useState } from "react";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { useStore } from "@/store/useStore";

interface CbRow {
  rank: number;
  ticker: string;
  price: number;
  rangeHigh: number;
  rangeLow: number;
  rangePct: number;
  type: "CONSOLIDATING" | "BULLISH BREAKOUT" | "BEARISH BREAKOUT";
  breakoutPct: number;
  slope: number;
  strength: number;
  volRatio: number;
}

export default function ConsolidationBreakoutPanel() {
  const { setSelectedStock, setFocusPattern, setShowChartModal } = useStore();
  const [data, setData] = useState<CbRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingStock, setLoadingStock] = useState<string | null>(null);
  const [filter, setFilter] = useState<"ALL" | "CONSOLIDATING" | "BULLISH BREAKOUT" | "BEARISH BREAKOUT">("ALL");

  useEffect(() => {
    setLoading(true);
    fetch("/api/scanner/consolidation-breakout")
      .then((r) => r.json())
      .then((d) => setData(d.stocks || []))
      .catch(() => setData([]))
      .finally(() => setLoading(false));
  }, []);

  const handleViewChart = async (row: CbRow) => {
    setLoadingStock(row.ticker);
    setSelectedStock(row.ticker);
    try {
      const res = await fetch(`/api/stock/${row.ticker}`);
      if (!res.ok) throw new Error("Not found");
      const data = await res.json();
      if (data.patterns && data.patterns.length > 0) {
        setFocusPattern(data.patterns[data.patterns.length - 1]);
      } else {
        setFocusPattern(null);
      }
      setShowChartModal(true);
    } catch {
      setFocusPattern(null);
      setShowChartModal(true);
    } finally {
      setLoadingStock(null);
    }
  };

  const filtered = filter === "ALL" ? data : data.filter((r) => r.type === filter);
  const consolCount = data.filter((r) => r.type === "CONSOLIDATING").length;
  const bullCount = data.filter((r) => r.type === "BULLISH BREAKOUT").length;
  const bearCount = data.filter((r) => r.type === "BEARISH BREAKOUT").length;

  const typeColor = (t: string) => {
    switch (t) {
      case "BULLISH BREAKOUT": return "text-[#22C55E]";
      case "BEARISH BREAKOUT": return "text-[#EF4444]";
      default: return "text-[#F59E0B]";
    }
  };

  const typeBadge = (t: string) => {
    switch (t) {
      case "BULLISH BREAKOUT": return "▲ BULL";
      case "BEARISH BREAKOUT": return "▼ BEAR";
      default: return "— SIDE";
    }
  };

  return (
    <div className="panel h-full flex flex-col">
      <div className="panel-header shrink-0 drag-handle cursor-grab">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold">CONSOLIDATION / BREAKOUT</span>
          <span className="text-[10px] text-[#4B5563]">{data.length} stocks</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex rounded-lg overflow-hidden glass text-[10px]">
            <button
              className={`px-2 py-1 font-medium ${filter === "ALL" ? "bg-white/10 text-white" : "text-[#94A3B8] hover:text-white"}`}
              onClick={() => setFilter("ALL")}
            >All ({data.length})</button>
            <button
              className={`px-2 py-1 font-medium flex items-center gap-1 ${filter === "CONSOLIDATING" ? "bg-[#F59E0B]/20 text-[#F59E0B]" : "text-[#94A3B8] hover:text-white"}`}
              onClick={() => setFilter("CONSOLIDATING")}
            ><Minus size={10} />{consolCount}</button>
            <button
              className={`px-2 py-1 font-medium flex items-center gap-1 ${filter === "BULLISH BREAKOUT" ? "bg-[#22C55E]/20 text-[#22C55E]" : "text-[#94A3B8] hover:text-white"}`}
              onClick={() => setFilter("BULLISH BREAKOUT")}
            ><TrendingUp size={10} />{bullCount}</button>
            <button
              className={`px-2 py-1 font-medium flex items-center gap-1 ${filter === "BEARISH BREAKOUT" ? "bg-[#EF4444]/20 text-[#EF4444]" : "text-[#94A3B8] hover:text-white"}`}
              onClick={() => setFilter("BEARISH BREAKOUT")}
            ><TrendingDown size={10} />{bearCount}</button>
          </div>
          <button
            className="btn-ghost text-[10px] text-[#3B82F6]"
            onClick={() => { setLoading(true); fetch("/api/scanner/consolidation-breakout").then((r) => r.json()).then((d) => setData(d.stocks || [])).catch(() => setData([])).finally(() => setLoading(false)); }}
          >Refresh</button>
        </div>
      </div>
      <div className="flex-1 overflow-auto">
        {loading && (
          <div className="flex items-center justify-center h-full text-[#4B5563] text-xs">Loading...</div>
        )}
        {!loading && filtered.length === 0 && (
          <div className="flex items-center justify-center h-full text-[#4B5563] text-xs">No stocks found</div>
        )}
        {!loading && filtered.length > 0 && (
          <table className="w-full text-[11px]">
            <thead>
              <tr className="text-[#4B5563] border-b border-white/5 sticky top-0 bg-[#0B0F17]">
                <th className="text-left px-3 py-2 font-medium">#</th>
                <th className="text-left px-3 py-2 font-medium">Ticker</th>
                <th className="text-right px-3 py-2 font-medium">Price</th>
                <th className="text-right px-3 py-2 font-medium">Range%</th>
                <th className="text-right px-3 py-2 font-medium">Type</th>
                <th className="text-right px-3 py-2 font-medium">Strength</th>
                <th className="text-right px-3 py-2 font-medium">Vol/Break</th>
              </tr>
            </thead>
            <tbody>
              {filtered.slice(0, 50).map((r) => (
                <tr
                  key={r.ticker}
                  className="border-b border-white/5 hover:bg-white/[0.04] transition-colors cursor-pointer"
                  onClick={() => handleViewChart(r)}
                >
                  <td className="px-3 py-2 text-[#4B5563]">{r.rank}</td>
                  <td className="px-3 py-2 font-medium text-white flex items-center gap-1">
                    {r.ticker}
                    {loadingStock === r.ticker && <span className="w-2 h-2 rounded-full bg-[#3B82F6] animate-pulse" />}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">{r.price.toFixed(2)}</td>
                  <td className="px-3 py-2 text-right tabular-nums text-[#94A3B8]">{r.rangePct.toFixed(1)}%</td>
                  <td className={`px-3 py-2 text-right font-medium ${typeColor(r.type)}`}>
                    {typeBadge(r.type)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-[#A78BFA]">{r.strength}</td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {r.type !== "CONSOLIDATING"
                      ? <span className="text-[#F59E0B]">{r.breakoutPct.toFixed(1)}%</span>
                      : <span className="text-[#4B5563]">{r.volRatio.toFixed(1)}x</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}