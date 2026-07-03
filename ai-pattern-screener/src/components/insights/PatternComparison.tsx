"use client";

import { useStore } from "@/store/useStore";
import { GitCompare, Activity } from "lucide-react";
import { useEffect, useState } from "react";
import { indianStocks } from "@/lib/utils";

export default function PatternComparison() {
  const { selectedStock, compareStock, setCompareStock, focusPattern } = useStore();
  const [showDropdown, setShowDropdown] = useState(false);
  const [comparePattern, setComparePattern] = useState<any>(null);

  useEffect(() => {
    if (!compareStock || compareStock === selectedStock) return;
    fetch(`/api/stock/${compareStock}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.patterns?.length > 0) {
          setComparePattern(data.patterns[data.patterns.length - 1]);
        } else {
          setComparePattern(null);
        }
      })
      .catch(() => setComparePattern(null));
  }, [compareStock, selectedStock]);

  const mainPattern = focusPattern;

  return (
    <div className="panel h-full flex flex-col">
      <div className="panel-header shrink-0 drag-handle cursor-grab active:cursor-grabbing">
        <div className="flex items-center gap-2">
          <GitCompare size={12} className="text-[#8B5CF6]" />
          <span className="text-xs font-semibold">PATTERN COMPARISON</span>
        </div>
        <div className="relative">
          <button
            onClick={() => setShowDropdown(!showDropdown)}
            className="text-[10px] btn-ghost flex items-center gap-1"
          >
            <span className="text-[#4B5563]">vs</span>
            <span className="text-white">{compareStock}</span>
          </button>
          {showDropdown && (
            <div className="absolute top-full right-0 mt-1 w-32 panel p-1 z-50 max-h-40 overflow-y-auto">
              {indianStocks.slice(0, 20).map((s) => (
                <button
                  key={s}
                  className={`w-full text-left px-2 py-1 text-[10px] rounded glass-hover ${s === compareStock ? "text-[#3B82F6]" : "text-[#94A3B8]"}`}
                  onClick={() => { setCompareStock(s); setShowDropdown(false); }}
                >
                  {s}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 p-3 space-y-3 overflow-y-auto">
        {!mainPattern ? (
          <div className="flex items-center justify-center h-full text-[#4B5563] text-xs">
            Click a stock row to see pattern comparison
          </div>
        ) : (
          <>
            <div className="text-center glass rounded-lg p-3">
              <div className="text-[10px] text-[#4B5563] mb-1">Pattern Similarity</div>
              <div className="text-2xl font-bold text-[#8B5CF6]">
                {comparePattern ? `${Math.round((1 - Math.abs(mainPattern.bodyVsAvg - (comparePattern.bodyVsAvg || 0)) / 5) * 100)}%` : "—"}
              </div>
              <div className="text-[10px] text-[#22C55E] mt-0.5">
                {comparePattern ? `${selectedStock} vs ${compareStock}` : "Select comparison stock"}
              </div>
            </div>

            <div className="glass rounded-lg h-24 flex items-center justify-center" style={{ border: "1px dashed rgba(255,255,255,0.08)" }}>
              <div className="text-[10px] text-[#4B5563] text-center px-4">
                {selectedStock} — BCC {mainPattern.triggerType} ({mainPattern.triggerDate})
                {comparePattern ? ` | ${compareStock} — BCC ${comparePattern.triggerType} (${comparePattern.triggerDate})` : ""}
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-[10px] font-medium text-[#4B5563] uppercase tracking-wider">Feature Comparison</div>
              {[
                { label: "Body Ratio", ref: `${mainPattern.bodyVsAvg.toFixed(1)}x`, match: comparePattern ? `${comparePattern.bodyVsAvg.toFixed(1)}x` : "—", diff: comparePattern ? `${Math.abs(mainPattern.bodyVsAvg - comparePattern.bodyVsAvg).toFixed(1)}x` : "—" },
                { label: "Volume Ratio", ref: `${mainPattern.volVsAvg.toFixed(1)}x`, match: comparePattern ? `${comparePattern.volVsAvg.toFixed(1)}x` : "—", diff: comparePattern ? `${Math.abs(mainPattern.volVsAvg - comparePattern.volVsAvg).toFixed(1)}x` : "—" },
                { label: "Consolidation", ref: `${mainPattern.consolCount}d`, match: comparePattern ? `${comparePattern.consolCount}d` : "—", diff: comparePattern ? `${Math.abs(mainPattern.consolCount - comparePattern.consolCount)}d` : "—" },
                { label: "RSI", ref: `${mainPattern.rsi}`, match: comparePattern ? `${comparePattern.rsi}` : "—", diff: comparePattern ? `${Math.abs(mainPattern.rsi - comparePattern.rsi)}` : "—" },
                { label: "Status", ref: mainPattern.status, match: comparePattern ? comparePattern.status : "—", diff: mainPattern.status === comparePattern?.status ? "Match" : "Diff" },
              ].map((f) => (
                <div key={f.label} className="flex items-center justify-between glass rounded-lg px-3 py-2">
                  <div>
                    <div className="text-[10px] text-[#94A3B8]">{f.label}</div>
                    <div className="text-xs text-white">{f.ref} → {f.match}</div>
                  </div>
                  <span className={`text-[10px] font-medium ${f.diff === "Match" ? "text-[#22C55E]" : "text-[#94A3B8]"}`}>{f.diff}</span>
                </div>
              ))}
            </div>

            {comparePattern && (
              <div className="glass rounded-lg p-2.5 border-l-2" style={{ borderLeftColor: "#8B5CF6" }}>
                <div className="flex items-center gap-1.5 mb-1">
                  <Activity size={10} className="text-[#8B5CF6]" />
                  <span className="text-[10px] text-[#8B5CF6] font-medium">AI Analysis</span>
                </div>
                <div className="text-[11px] text-[#94A3B8] leading-relaxed">
                  {mainPattern.triggerType === comparePattern.triggerType
                    ? `Both ${selectedStock} and ${compareStock} show ${mainPattern.triggerType} BCC patterns. Similar structure with ${mainPattern.consolCount}d consolidation vs ${comparePattern.consolCount}d.`
                    : `${selectedStock} shows ${mainPattern.triggerType} pattern while ${compareStock} shows ${comparePattern.triggerType} — opposite directional bias.`}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
