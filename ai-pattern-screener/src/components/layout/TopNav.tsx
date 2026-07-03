"use client";

import { useStore } from "@/store/useStore";
import {
  Search, Bell, Settings, Zap, Pause, Save, Download, Upload,
  Clock, Wifi, BarChart3, ChevronDown, LayoutDashboard, Radio, Database, RotateCcw,
} from "lucide-react";
import { indianStocks, timeframes, patternLengths } from "@/lib/utils";
import { useState, useRef, useEffect } from "react";

export default function TopNav() {
  const {
    selectedStock, setSelectedStock, selectedTimeframe, setSelectedTimeframe,
    patternLength, setPatternLength, similarityThreshold, setSimilarityThreshold,
    scanning, setScanning, globalSearch, setGlobalSearch, unreadAlerts,
    toggleSidebar, fetchScanner, fetchLiveScanner, scannerData, liveScanning, scannerLoading,
    cacheStatus, cacheRefreshing, fetchCacheStatus, refreshCache, forceRefreshCache, forceRefreshing,
  } = useStore();

  const [searchFocused, setSearchFocused] = useState(false);

  useEffect(() => { fetchCacheStatus(); }, [fetchCacheStatus]);
  const [showStockDropdown, setShowStockDropdown] = useState(false);
  const [showTFDropdown, setShowTFDropdown] = useState(false);
  const [showPLDropdown, setShowPLDropdown] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);

  const filteredStocks = indianStocks.filter((s) =>
    s.toLowerCase().includes(globalSearch.toLowerCase())
  ).slice(0, 8);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setSearchFocused(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handleScanToggle = () => {
    const newState = !scanning;
    setScanning(newState);
    if (newState) {
      // Quick scan from CSV data first
      fetchScanner();
    }
  };

  return (
    <nav className="h-16 flex items-center justify-between px-5 border-b bg-[#0B0F17]/95 backdrop-blur-xl" style={{ borderColor: "rgba(255,255,255,0.08)" }}>
      <div className="flex items-center gap-4">
        <button onClick={toggleSidebar} className="btn-ghost p-1.5 -ml-1.5">
          <BarChart3 size={18} className="text-[#3B82F6]" />
        </button>
        <div className="flex items-center gap-1.5">
          <span className="text-sm font-semibold tracking-tight">APEX</span>
          <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-[#3B82F6]/15 text-[#3B82F6]">AI</span>
        </div>
        <div className="w-px h-5" style={{ background: "rgba(255,255,255,0.08)" }} />

        <div className="relative" ref={searchRef}>
          <div className="flex items-center gap-2 input w-72" style={{ cursor: "text" }} onClick={() => searchRef.current?.focus()}>
            <Search size={14} className="text-[#4B5563]" />
            <input
              ref={searchRef}
              value={globalSearch}
              onChange={(e) => setGlobalSearch(e.target.value)}
              onFocus={() => setSearchFocused(true)}
              placeholder="Search ticker, pattern, sector..."
              className="bg-transparent border-none text-sm text-white flex-1 outline-none placeholder:text-[#4B5563]"
            />
            <kbd className="text-[10px] px-1.5 py-0.5 rounded bg-white/5 text-[#4B5563]">⌘K</kbd>
          </div>
          {searchFocused && globalSearch && (
            <div className="absolute top-full mt-1 w-72 panel p-1 z-50">
              {filteredStocks.map((s) => (
                <button
                  key={s}
                  className="w-full text-left px-3 py-2 text-sm rounded-lg glass-hover flex items-center justify-between"
                  onClick={() => { setSelectedStock(s); setGlobalSearch(""); setSearchFocused(false); }}
                >
                  <span>{s}</span>
                  <span className="text-[11px] text-[#94A3B8]">Stock</span>
                </button>
              ))}
              {filteredStocks.length === 0 && (
                <div className="px-3 py-4 text-xs text-center text-[#4B5563]">No results</div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="flex items-center gap-2">
        <div className="relative">
          <button
            onClick={() => setShowStockDropdown(!showStockDropdown)}
            className="btn-glass flex items-center gap-2 min-w-[120px]"
          >
            <span className="status-dot live" />
            <span className="text-sm">{selectedStock}</span>
            <ChevronDown size={12} className="text-[#94A3B8]" />
          </button>
          {showStockDropdown && (
            <div className="absolute top-full mt-1 left-0 w-40 panel p-1 z-50 max-h-60 overflow-y-auto">
              {indianStocks.slice(0, 30).map((s) => (
                <button
                  key={s}
                  className="w-full text-left px-3 py-1.5 text-xs rounded-lg glass-hover"
                  onClick={() => { setSelectedStock(s); setShowStockDropdown(false); }}
                >
                  {s}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="relative">
          <button
            onClick={() => setShowTFDropdown(!showTFDropdown)}
            className="btn-glass flex items-center gap-2"
          >
            <Clock size={12} className="text-[#94A3B8]" />
            <span className="text-sm">{selectedTimeframe}</span>
            <ChevronDown size={10} className="text-[#94A3B8]" />
          </button>
          {showTFDropdown && (
            <div className="absolute top-full mt-1 left-0 w-20 panel p-1 z-50">
              {timeframes.map((tf) => (
                <button
                  key={tf}
                  className={`w-full text-left px-3 py-1.5 text-xs rounded-lg glass-hover ${tf === selectedTimeframe ? "text-[#3B82F6]" : ""}`}
                  onClick={() => { setSelectedTimeframe(tf); setShowTFDropdown(false); }}
                >
                  {tf}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="relative">
          <button
            onClick={() => setShowPLDropdown(!showPLDropdown)}
            className="btn-glass flex items-center gap-2"
          >
            <span className="text-xs text-[#94A3B8]">Len</span>
            <span className="text-sm">{patternLength}</span>
            <ChevronDown size={10} className="text-[#94A3B8]" />
          </button>
          {showPLDropdown && (
            <div className="absolute top-full mt-1 left-0 w-20 panel p-1 z-50">
              {patternLengths.map((pl) => (
                <button
                  key={pl}
                  className={`w-full text-left px-3 py-1.5 text-xs rounded-lg glass-hover ${pl === patternLength ? "text-[#3B82F6]" : ""}`}
                  onClick={() => { setPatternLength(pl); setShowPLDropdown(false); }}
                >
                  {pl}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2 glass rounded-lg px-3 py-1.5">
          <span className="text-xs text-[#94A3B8]">Sim</span>
          <input
            type="range"
            min={50}
            max={99}
            value={similarityThreshold}
            onChange={(e) => setSimilarityThreshold(Number(e.target.value))}
            className="w-20 h-1 accent-[#3B82F6] cursor-pointer"
          />
          <span className="text-xs font-medium text-[#3B82F6] w-7 text-right">{similarityThreshold}%</span>
        </div>

        <div className="w-px h-5" style={{ background: "rgba(255,255,255,0.08)" }} />

        <button
          onClick={handleScanToggle}
          className={`btn flex items-center gap-1.5 ${scanning ? "bg-[#22C55E]/15 text-[#22C55E]" : "bg-[#3B82F6] text-white hover:bg-[#2563EB]"}`}
        >
          {scanning ? <Pause size={13} /> : <Zap size={13} />}
          <span>{scanning ? "Pause" : "Scan"}</span>
        </button>
        <button
          onClick={() => fetchLiveScanner()}
          disabled={liveScanning || scannerLoading}
          className="btn flex items-center gap-1.5 bg-[#8B5CF6]/15 text-[#8B5CF6] hover:bg-[#8B5CF6]/25 disabled:opacity-40"
          title="Live scan via Dhan API (uses cached data if available)"
        >
          <Radio size={13} className={liveScanning ? "animate-pulse" : ""} />
          <span>{liveScanning ? "Scanning..." : "Live"}</span>
        </button>
        <button
          onClick={() => refreshCache()}
          disabled={cacheRefreshing}
          className="btn flex items-center gap-1.5 glass text-[#94A3B8] hover:text-white disabled:opacity-40"
          title={`Dhan cache: ${cacheStatus ? `${cacheStatus.fresh}/${cacheStatus.total} stocks fresh` : "N/A"}`}
        >
          <Database size={12} className={cacheRefreshing ? "animate-spin" : ""} />
          <span className="text-[10px]">{cacheStatus ? `${cacheStatus.fresh}/${cacheStatus.total}` : "Cache"}</span>
          <RotateCcw size={10} className="text-[#4B5563]" />
        </button>
        <button
          onClick={() => forceRefreshCache()}
          disabled={forceRefreshing}
          className="btn flex items-center gap-1.5 bg-[#3B82F6]/15 text-[#3B82F6] hover:bg-[#3B82F6]/25 disabled:opacity-40"
          title="Fetch last 30 days of live data for all stocks from Dhan API, then run scanner"
        >
          <Database size={12} className={forceRefreshing ? "animate-spin" : ""} />
          <span className="text-[10px]">{forceRefreshing ? `Fetching ${cacheStatus?.total || 487}...` : "Fetch Data"}</span>
        </button>
        <button className="btn-ghost p-1.5"><Save size={14} /></button>
        <button className="btn-ghost p-1.5"><Download size={14} /></button>
        <button className="btn-ghost p-1.5"><Upload size={14} /></button>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-xs text-[#94A3B8]">
          <div className="flex items-center gap-1.5">
            <Wifi size={10} className="text-[#22C55E]" />
            <span>Live</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="status-dot live" />
            <span>{scannerData.length > 0 ? `${scannerData.length} stocks` : "0.3ms"}</span>
          </div>
        </div>
        <div className="w-px h-5" style={{ background: "rgba(255,255,255,0.08)" }} />
        <button className="relative btn-ghost p-1.5">
          <Bell size={15} />
          {unreadAlerts > 0 && (
            <span className="absolute -top-0.5 -right-0.5 w-4 h-4 rounded-full bg-[#EF4444] text-[9px] font-bold flex items-center justify-center">
              {unreadAlerts}
            </span>
          )}
        </button>
        <button className="btn-ghost p-1.5"><Settings size={15} /></button>
        <div className="flex items-center gap-2 glass rounded-lg px-2.5 py-1.5">
          <div className="w-5 h-5 rounded-full bg-gradient-to-br from-[#3B82F6] to-[#8B5CF6] flex items-center justify-center text-[9px] font-bold">A</div>
          <span className="text-sm">Admin</span>
        </div>
      </div>
    </nav>
  );
}
