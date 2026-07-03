"use client";

import { useStore, type PatternHighlight } from "@/store/useStore";
import { formatVolume, formatPercent } from "@/lib/utils";
import {
  useReactTable, getCoreRowModel, getSortedRowModel, flexRender,
  createColumnHelper, type SortingState,
} from "@tanstack/react-table";
import { useState, useMemo, useEffect } from "react";
import { ArrowUpDown, AlertCircle, ChartNoAxesColumnIncreasing, Loader2, Radio, Clock, AlertTriangle } from "lucide-react";

const columnHelper = createColumnHelper<any>();

export default function ScannerTable() {
  const {
    scannerData, setSelectedStock, setFocusPattern, setShowChartModal,
    fetchScanner, scanning, setScannerMode, fetchAIInsight, scannerMode,
    fetchHistoryScanner, liveScanning, historyLoading, scannerLoading,
  } = useStore();
  const [sorting, setSorting] = useState<SortingState>([{ id: "aiScore", desc: true }]);
  const [loadingStock, setLoadingStock] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const init = async () => {
      setError(null);
      try {
        if (scannerMode === "live") await fetchScanner();
        else await fetchHistoryScanner();
      } catch {
        setError(`Failed to load ${scannerMode} data`);
      }
    };
    init();
  }, []);

  useEffect(() => {
    if (!scanning || scannerMode !== "live") return;
    const interval = setInterval(() => fetchScanner(), 30000);
    return () => clearInterval(interval);
  }, [scanning, scannerMode, fetchScanner]);

  const handleViewChart = async (row: any) => {
    const sym = row.ticker;
    setLoadingStock(sym);
    setSelectedStock(sym);

    try {
      const res = await fetch(`/api/stock/${sym}`);
      if (!res.ok) throw new Error("Not found");
      const data = await res.json();

      if (data.patterns && data.patterns.length > 0) {
        const latest = data.patterns[data.patterns.length - 1] as PatternHighlight;
        setFocusPattern(latest);
        setShowChartModal(true);
        fetchAIInsight(sym);
      } else {
        setFocusPattern(null);
        setShowChartModal(true);
      }
    } catch {
      setFocusPattern(null);
      setShowChartModal(true);
    } finally {
      setLoadingStock(null);
    }
  };

  const columns = useMemo(
    () => [
      columnHelper.accessor("rank", {
        header: "#",
        cell: (info) => <span className="text-[#4B5563] text-xs">{info.getValue()}</span>,
        size: 40,
      }),
      columnHelper.accessor("ticker", {
        header: "Ticker",
        cell: (info) => {
          const ticker = info.getValue() as string;
          return (
            <div className="flex items-center gap-2">
              <button
                onClick={(e) => { e.stopPropagation(); handleViewChart(info.row.original); }}
                className="font-semibold text-sm hover:text-[#3B82F6] transition-colors text-white"
              >
                {ticker}
              </button>
              <span className={`text-[9px] px-1 py-0.5 rounded ${scannerMode === "live" ? "bg-[#22C55E]/15 text-[#22C55E]" : "bg-[#8B5CF6]/15 text-[#8B5CF6]"}`}>
                {scannerMode === "live" ? "LIVE" : "HIST"}
              </span>
            </div>
          );
        },
        size: 120,
      }),
      columnHelper.accessor("price", {
        header: "Price",
        cell: (info) => <span className="text-sm tabular-nums">{info.getValue()?.toLocaleString() || "—"}</span>,
        size: 90,
      }),
      columnHelper.accessor("change", {
        header: "Chg %",
        cell: (info) => {
          const v = info.getValue();
          return (
            <span className={`text-sm tabular-nums font-medium ${v >= 0 ? "text-[#22C55E]" : "text-[#EF4444]"}`}>
              {v != null ? formatPercent(v) : "—"}
            </span>
          );
        },
        size: 85,
      }),
      columnHelper.accessor("pattern", {
        header: "Pattern",
        cell: (info) => {
          const v = info.getValue() || "";
          return (
            <span className={`text-xs font-medium truncate max-w-[140px] block ${
              v.includes("BCC") ? "text-[#8B5CF6]" : v.includes("SQUEEZE") ? "text-[#3B82F6]" : "text-[#4B5563]"
            }`}>{v}</span>
          );
        },
        size: 150,
      }),
      ...(scannerMode === "history"
        ? [
            columnHelper.accessor("patternCount", {
              header: "Total",
              cell: (info) => (
                <span className="text-xs font-bold text-[#8B5CF6]">{info.getValue()}</span>
              ),
              size: 55,
            }),
            columnHelper.accessor("successRate", {
              header: "W%",
              cell: (info) => (
                <span className={`text-xs font-medium ${info.getValue() > 50 ? "text-[#22C55E]" : "text-[#EF4444]"}`}>
                  {info.getValue()}%
                </span>
              ),
              size: 45,
            }),
            columnHelper.accessor("latestDate", {
              header: "Latest",
              cell: (info) => <span className="text-[10px] text-[#94A3B8]">{info.getValue()}</span>,
              size: 85,
            }),
            columnHelper.accessor("latestStatus", {
              header: "Status",
              cell: (info) => {
                const v = info.getValue();
                return (
                  <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                    v === "BROKEN UP" ? "bg-[#22C55E]/15 text-[#22C55E]" :
                    v === "BROKEN DOWN" ? "bg-[#EF4444]/15 text-[#EF4444]" :
                    "bg-[#F59E0B]/15 text-[#F59E0B]"
                  }`}>{v}</span>
                );
              },
              size: 85,
            }),
          ]
        : [
            columnHelper.accessor("aiScore", {
              header: "AI",
              cell: (info) => (
                <div className="flex items-center gap-1.5">
                  <div className="w-14 h-1.5 rounded-full bg-white/5 overflow-hidden">
                    <div className="h-full rounded-full bg-[#3B82F6]" style={{ width: `${info.getValue() || 0}%` }} />
                  </div>
                  <span className="text-xs text-[#3B82F6] tabular-nums w-8">{(info.getValue() || 0).toFixed(0)}</span>
                </div>
              ),
              size: 85,
            }),
            columnHelper.accessor("rsi", {
              header: "RSI",
              cell: (info) => {
                const v = info.getValue();
                return (
                  <span className={`text-xs tabular-nums ${v > 70 ? "text-[#EF4444]" : v < 30 ? "text-[#22C55E]" : "text-[#94A3B8]"}`}>
                    {v || "—"}
                  </span>
                );
              },
              size: 50,
            }),
            columnHelper.accessor("trend", {
              header: "Trend",
              cell: (info) => {
                const v = info.getValue();
                return (
                  <span className={`text-xs ${
                    v === "Strong Up" || v === "Up" ? "text-[#22C55E]" :
                    v === "Strong Down" || v === "Down" ? "text-[#EF4444]" : "text-[#94A3B8]"
                  }`}>{v || "—"}</span>
                );
              },
              size: 85,
            }),
          ]),
      columnHelper.display({
        id: "chart",
        header: "",
        cell: (info) => (
          <button
            onClick={(e) => { e.stopPropagation(); handleViewChart(info.row.original); }}
            className="btn-ghost p-1 text-[#4B5563] hover:text-[#3B82F6]"
            title="View chart"
          >
            {loadingStock === info.row.original.ticker ? (
              <Loader2 size={12} className="animate-spin" />
            ) : (
              <ChartNoAxesColumnIncreasing size={12} />
            )}
          </button>
        ),
        size: 35,
      }),
    ],
    [loadingStock, scannerMode]
  );

  const table = useReactTable({
    data: scannerData,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const handleModeSwitch = async (mode: "live" | "history") => {
    setError(null);
    setScannerMode(mode);
    try {
      if (mode === "live") {
        await fetchScanner();
      } else {
        await fetchHistoryScanner();
      }
    } catch {
      setError(`Failed to load ${mode} data`);
    }
  };

  const isLoading = scannerLoading || liveScanning || historyLoading;

  return (
    <div className="panel h-full flex flex-col overflow-hidden">
      <div className="panel-header shrink-0 drag-handle cursor-grab active:cursor-grabbing">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold">PATTERN SCANNER</span>
          <span className="text-[10px] text-[#4B5563]">{scannerData.length} stocks</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="flex rounded-lg overflow-hidden glass text-[10px]">
            <button
              onClick={() => handleModeSwitch("live")}
              className={`px-2.5 py-1 font-medium transition-colors flex items-center gap-1 ${
                scannerMode === "live" ? "bg-[#3B82F6]/20 text-[#3B82F6]" : "text-[#94A3B8] hover:text-white"
              }`}
            >
              <Radio size={10} /> Live (10c)
            </button>
            <div className="w-px bg-white/5" />
            <button
              onClick={() => handleModeSwitch("history")}
              className={`px-2.5 py-1 font-medium transition-colors flex items-center gap-1 ${
                scannerMode === "history" ? "bg-[#8B5CF6]/20 text-[#8B5CF6]" : "text-[#94A3B8] hover:text-white"
              }`}
            >
              <Clock size={10} /> History
            </button>
          </div>
          {isLoading && <Loader2 size={10} className="animate-spin text-[#3B82F6]" />}
          <span className={`status-dot ${scanning ? "live" : ""}`} />
        </div>
      </div>
      <div className="flex-1 overflow-auto">
        {error ? (
          <div className="flex items-center justify-center h-full text-[#EF4444] text-sm">
            <div className="text-center space-y-2">
              <AlertTriangle size={20} className="mx-auto" />
              <p>{error}</p>
              <button onClick={() => handleModeSwitch(scannerMode)} className="btn-ghost text-xs text-[#3B82F6]">
                Retry
              </button>
            </div>
          </div>
        ) : scannerData.length === 0 && !isLoading ? (
          <div className="flex items-center justify-center h-full text-[#4B5563] text-sm">
            <div className="text-center space-y-2">
              <p>No patterns found in recent candles</p>
              <button onClick={() => handleModeSwitch(scannerMode)} className="btn-ghost text-xs text-[#3B82F6]">
                {scannerMode === "live" ? "Rescan" : "Reload"}
              </button>
            </div>
          </div>
        ) : scannerData.length === 0 && isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 size={24} className="animate-spin text-[#3B82F6]" />
          </div>
        ) : (
          <table className="w-full">
            <thead className="sticky top-0 bg-[#0B0F17]/95 backdrop-blur-xl z-10">
              {table.getHeaderGroups().map((hg) => (
                <tr key={hg.id}>
                  {hg.headers.map((h) => (
                    <th
                      key={h.id}
                      onClick={h.column.getToggleSortingHandler()}
                      className="text-left text-[10px] font-medium text-[#4B5563] px-3 py-2.5 cursor-pointer hover:text-[#94A3B8] whitespace-nowrap"
                      style={{ width: h.getSize() }}
                    >
                      <div className="flex items-center gap-1">
                        {flexRender(h.column.columnDef.header, h.getContext())}
                        <ArrowUpDown size={9} className="opacity-50" />
                      </div>
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.map((row, i) => (
                <tr
                  key={row.id}
                  className="group border-transparent hover:bg-white/[0.02] transition-colors cursor-pointer"
                  style={{ borderTop: i > 0 ? "1px solid rgba(255,255,255,0.03)" : undefined }}
                  onClick={() => handleViewChart(row.original)}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="px-3 py-2.5" style={{ width: cell.column.getSize() }}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
