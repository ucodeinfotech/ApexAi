"use client";

import { useEffect, useState } from "react";
import { TrendingUp, TrendingDown } from "lucide-react";
import { useStore } from "@/store/useStore";

interface DemaAtrRow {
  rank: number;
  ticker: string;
  price: number;
  demaAtr: number;
  direction: "UP" | "DOWN";
  slope: number;
  slope5: number;
  priceDistPct: number;
  volume: number;
}

export default function DemaAtrPanel() {
  const { setSelectedStock, setFocusPattern, setShowChartModal } = useStore();
  const [data, setData] = useState<DemaAtrRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingStock, setLoadingStock] = useState<string | null>(null);
  const [filter, setFilter] = useState<"ALL" | "UP" | "DOWN">("ALL");

  useEffect(() => {
    setLoading(true);
    fetch("/api/scanner/dema-atr")
      .then((r) => r.json())
      .then((d) => setData(d.stocks || []))
      .catch(() => setData([]))
      .finally(() => setLoading(false));
  }, []);

  const handleViewChart = async (row: DemaAtrRow) => {
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

  const filtered = filter === "ALL" ? data : data.filter((r) => r.direction === filter);
  const upCount = data.filter((r) => r.direction === "UP").length;
  const downCount = data.filter((r) => r.direction === "DOWN").length;

  return (
    <div className="panel h-full flex flex-col">
      <div className="panel-header shrink-0 drag-handle cursor-grab">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold">DEMA ATR SCANNER</span>
          <span className="text-[10px] text-[#4B5563]">{data.length} stocks</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex rounded-lg overflow-hidden glass text-[10px]">
            <button
              className={`px-2 py-1 font-medium ${filter === "ALL" ? "bg-white/10 text-white" : "text-[#94A3B8] hover:text-white"}`}
              onClick={() => setFilter("ALL")}
            >All ({data.length})</button>
            <button
              className={`px-2 py-1 font-medium flex items-center gap-1 ${filter === "UP" ? "bg-[#22C55E]/20 text-[#22C55E]" : "text-[#94A3B8] hover:text-white"}`}
              onClick={() => setFilter("UP")}
            ><TrendingUp size={10} />{upCount}</button>
            <button
              className={`px-2 py-1 font-medium flex items-center gap-1 ${filter === "DOWN" ? "bg-[#EF4444]/20 text-[#EF4444]" : "text-[#94A3B8] hover:text-white"}`}
              onClick={() => setFilter("DOWN")}
            ><TrendingDown size={10} />{downCount}</button>
          </div>
          <button
            className="btn-ghost text-[10px] text-[#3B82F6]"
            onClick={() => { setLoading(true); fetch("/api/scanner/dema-atr").then((r) => r.json()).then((d) => setData(d.stocks || [])).catch(() => setData([])).finally(() => setLoading(false)); }}
          >Refresh</button>
        </div>
      </div>
      <div className="flex-1 overflow-auto">
        {loading && (
          <div className="flex items-center justify-center h-full text-[#4B5563] text-xs">Loading...</div>
        )}
        {!loading && filtered.length === 0 && (
          <div className="flex items-center justify-center h-full text-[#4B5563] text-xs">No stocks near DEMA ATR line</div>
        )}
        {!loading && filtered.length > 0 && (
          <table className="w-full text-[11px]">
            <thead>
              <tr className="text-[#4B5563] border-b border-white/5 sticky top-0 bg-[#0B0F17]">
                <th className="text-left px-3 py-2 font-medium">#</th>
                <th className="text-left px-3 py-2 font-medium">Ticker</th>
                <th className="text-right px-3 py-2 font-medium">Price</th>
                <th className="text-right px-3 py-2 font-medium">DEMA ATR</th>
                <th className="text-right px-3 py-2 font-medium">Dir</th>
                <th className="text-right px-3 py-2 font-medium">Slope 5</th>
                <th className="text-right px-3 py-2 font-medium">Dist%</th>
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
                  <td className="px-3 py-2 text-right tabular-nums text-[#94A3B8]">{r.demaAtr.toFixed(2)}</td>
                  <td className="px-3 py-2 text-right">
                    <span className={`tag ${r.direction === "UP" ? "tag-bullish" : "tag-bearish"}`}>
                      {r.direction === "UP" ? "▲" : "▼"} {r.direction}
                    </span>
                  </td>
                  <td className={`px-3 py-2 text-right tabular-nums ${r.slope5 > 0 ? "text-[#22C55E]" : "text-[#EF4444]"}`}>
                    {r.slope5 > 0 ? "+" : ""}{r.slope5.toFixed(4)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-[#F59E0B]">
                    {r.priceDistPct.toFixed(1)}%
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
