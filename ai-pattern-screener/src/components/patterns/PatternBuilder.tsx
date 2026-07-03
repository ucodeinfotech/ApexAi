"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import { useStore } from "@/store/useStore";
import { createChart, ColorType, CrosshairMode, LineStyle } from "lightweight-charts";
import { indianStocks } from "@/lib/utils";
import { Plus, Save, Trash2, Sparkles, Check, X } from "lucide-react";

interface SavedPattern {
  id: string;
  name: string;
  category: string;
  stock: string;
  triggerDate: string;
  bodyVsAvg: number;
  volVsAvg: number;
  consolCount: number;
  createdAt: string;
}

const defaultParams = {
  bodyMult: 2.0,
  volMult: 1.5,
  wickPct: 0.2,
  consolMin: 3,
  consolBodyPct: 0.3,
  consolMaxRangePct: 0.05,
};

export default function PatternBuilder() {
  const chartRef = useRef<HTMLDivElement>(null);
  const [stock, setStock] = useState("RELIANCE");
  const [candleData, setCandleData] = useState<any[]>([]);
  const [patterns, setPatterns] = useState<any[]>([]);
  const [triggerIdx, setTriggerIdx] = useState<number | null>(null);
  const [triggerCandle, setTriggerCandle] = useState<any>(null);
  const [detectedPattern, setDetectedPattern] = useState<any>(null);
  const [params, setParams] = useState(defaultParams);
  const [name, setName] = useState("");
  const [category, setCategory] = useState("BCC Variant");
  const [savedPatterns, setSavedPatterns] = useState<SavedPattern[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("customPatterns");
    if (saved) setSavedPatterns(JSON.parse(saved));
  }, []);

  const fetchData = useCallback(async (sym: string) => {
    const res = await fetch(`/api/stock/${sym}`);
    if (!res.ok) return;
    const data = await res.json();
    const sorted = (data.candles || []).sort((a: any, b: any) => a.time - b.time);
    setCandleData(sorted);
    setPatterns(data.patterns || []);
    setTriggerIdx(null);
    setTriggerCandle(null);
    setDetectedPattern(null);
  }, []);

  useEffect(() => { fetchData(stock); }, [stock, fetchData]);

  const handleChartClick = useCallback((param: any) => {
    if (!param.time || candleData.length === 0) return;
    const idx = candleData.findIndex((d) => d.time === param.time);
    if (idx < 20) return;
    setTriggerIdx(idx);
    setTriggerCandle(candleData[idx]);

    const c = candleData[idx];
    const avgBody = c.avgBody20 || 0;
    const avgVol = c.avgVol20 || 0;
    if (!avgBody || !avgVol) { setDetectedPattern(null); return; }

    const wickRatio = (c.high - Math.max(c.close, c.open)) / c.range;
    let consolCount = 0, consolHigh = c.high, consolLow = c.low, consolEndIdx = -1;

    for (let j = idx + 1; j < candleData.length; j++) {
      const n = candleData[j];
      consolHigh = Math.max(consolHigh, n.high);
      consolLow = Math.min(consolLow, n.low);
      const rangePct = (consolHigh - consolLow) / c.close;
      if (rangePct > params.consolMaxRangePct) {
        if (consolCount >= params.consolMin) { consolEndIdx = j - 1; break; }
        consolCount = 0; consolHigh = n.high; consolLow = n.low; continue;
      }
      if ((n.body / n.range) < params.consolBodyPct) { consolCount++; }
      else {
        if (consolCount >= params.consolMin) { consolEndIdx = j - 1; break; }
        consolCount = 0; consolHigh = n.high; consolLow = n.low;
      }
    }
    if (consolCount >= params.consolMin && consolEndIdx < 0) consolEndIdx = candleData.length - 1;

    const lastCandle = consolEndIdx >= 0 ? candleData[consolEndIdx] : c;
    const changePct = ((lastCandle.close - c.close) / c.close) * 100;
    const triggerType = c.close > c.open ? "BULLISH" : "BEARISH";

    setDetectedPattern({
      triggerDate: new Date(c.time * 1000).toISOString().split("T")[0],
      triggerTime: c.time, triggerType,
      triggerClose: c.close,
      bodyVsAvg: Math.round(c.body / avgBody * 100) / 100,
      volVsAvg: Math.round(c.volume / avgVol * 100) / 100,
      wickPct: Math.round(wickRatio * 1000) / 10,
      rsi: Math.round(c.rsi14 || 50),
      consolCount, consolHigh, consolLow,
      consolRangePct: Math.round((consolHigh - consolLow) / c.close * 10000) / 100,
      consolEndTime: consolEndIdx >= 0 ? candleData[consolEndIdx].time : c.time,
      changePct: Math.round(changePct * 100) / 100,
      isTrigger: c.body >= avgBody * params.bodyMult && c.volume >= avgVol * params.volMult && wickRatio < params.wickPct && consolCount >= params.consolMin,
    });
  }, [candleData, params]);

  // Render chart
  useEffect(() => {
    if (!chartRef.current || candleData.length === 0) return;
    const chart = createChart(chartRef.current, {
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#4B5563", fontSize: 11 },
      grid: { vertLines: { color: "rgba(255,255,255,0.03)", style: LineStyle.Dashed }, horzLines: { color: "rgba(255,255,255,0.03)", style: LineStyle.Dashed } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "rgba(255,255,255,0.08)", scaleMargins: { top: 0.05, bottom: 0.25 } },
      timeScale: { borderColor: "rgba(255,255,255,0.08)", timeVisible: true, secondsVisible: false },
      handleScroll: true, handleScale: true,
    });

    const candleSeries = chart.addCandlestickSeries({ upColor: "#22C55E", downColor: "#EF4444", borderDownColor: "#EF4444", borderUpColor: "#22C55E", wickDownColor: "#EF4444", wickUpColor: "#22C55E" });
    candleSeries.setData(candleData);

    const volumeSeries = chart.addHistogramSeries({ priceFormat: { type: "volume" }, priceScaleId: "volume" });
    chart.priceScale("volume").applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
    volumeSeries.setData(candleData.map((d: any) => ({ time: d.time, value: d.volume, color: d.close >= d.open ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)" })));

    if (detectedPattern) {
      const mColor = detectedPattern.triggerType === "BULLISH" ? "#22C55E" : "#EF4444";
      candleSeries.setMarkers([{ time: detectedPattern.triggerTime, position: detectedPattern.triggerType === "BULLISH" ? "belowBar" as const : "aboveBar" as const, color: mColor, shape: detectedPattern.triggerType === "BULLISH" ? "arrowUp" as const : "arrowDown" as const, text: `BCC ${detectedPattern.triggerType}`, size: 1 }]);

      const tidx = candleData.findIndex((d) => d.time === detectedPattern.triggerTime);
      const eidx = candleData.findIndex((d) => d.time === detectedPattern.consolEndTime);
      if (tidx >= 0) {
        const lineStart = Math.max(0, tidx - 1);
        const lineEnd = Math.min(candleData.length - 1, eidx >= 0 ? eidx + 2 : candleData.length - 1);
        const hlSeries = chart.addLineSeries({ color: "rgba(139,92,246,0.4)", lineWidth: 1, lineStyle: LineStyle.Dashed, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
        const hData: any[] = [];
        for (let k = lineStart; k <= lineEnd; k++) hData.push({ time: candleData[k].time, value: detectedPattern.consolHigh });
        if (hData.length >= 2) hlSeries.setData(hData);

        const llSeries = chart.addLineSeries({ color: "rgba(139,92,246,0.4)", lineWidth: 1, lineStyle: LineStyle.Dashed, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
        const lData: any[] = [];
        for (let k = lineStart; k <= lineEnd; k++) lData.push({ time: candleData[k].time, value: detectedPattern.consolLow });
        if (lData.length >= 2) llSeries.setData(lData);
      }
    }

    chart.timeScale().fitContent();
    chart.subscribeClick(handleChartClick);

    const resize = () => { if (chartRef.current) chart.applyOptions({ width: chartRef.current.clientWidth, height: chartRef.current.clientHeight }); };
    window.addEventListener("resize", resize);
    setTimeout(resize, 100);
    return () => { window.removeEventListener("resize", resize); chart.remove(); };
  }, [candleData, detectedPattern, handleChartClick]);

  const savePattern = () => {
    if (!name.trim() || !detectedPattern) return;
    const newPattern: SavedPattern = {
      id: Date.now().toString(36),
      name: name.trim(),
      category,
      stock,
      triggerDate: detectedPattern.triggerDate,
      bodyVsAvg: detectedPattern.bodyVsAvg,
      volVsAvg: detectedPattern.volVsAvg,
      consolCount: detectedPattern.consolCount,
      createdAt: new Date().toISOString(),
    };
    const updated = [...savedPatterns, newPattern];
    setSavedPatterns(updated);
    localStorage.setItem("customPatterns", JSON.stringify(updated));
    setName("");
  };

  const deletePattern = (id: string) => {
    const updated = savedPatterns.filter((p) => p.id !== id);
    setSavedPatterns(updated);
    localStorage.setItem("customPatterns", JSON.stringify(updated));
  };

  const paramsList = [
    { key: "bodyMult", label: "Body Multiplier", step: 0.1, min: 1, max: 5 },
    { key: "volMult", label: "Vol Multiplier", step: 0.1, min: 1, max: 5 },
    { key: "wickPct", label: "Max Wick %", step: 0.05, min: 0, max: 0.5 },
    { key: "consolMin", label: "Min Consol Candles", step: 1, min: 1, max: 10 },
    { key: "consolBodyPct", label: "Consol Body/Range", step: 0.05, min: 0.1, max: 0.5 },
    { key: "consolMaxRangePct", label: "Max Consol Range %", step: 0.01, min: 0.01, max: 0.15 },
  ];

  return (
    <div className="panel h-full flex flex-col">
      <div className="panel-header shrink-0 drag-handle cursor-grab active:cursor-grabbing">
        <div className="flex items-center gap-2">
          <Sparkles size={12} className="text-[#F59E0B]" />
          <span className="text-xs font-semibold">PATTERN BUILDER</span>
        </div>
        <div className="relative">
          <button onClick={() => setShowDropdown(!showDropdown)} className="text-[10px] btn-ghost px-2 py-0.5">
            {stock} <span className="text-[#4B5563]">▼</span>
          </button>
          {showDropdown && (
            <div className="absolute top-full right-0 mt-1 w-28 panel p-1 z-50 max-h-40 overflow-y-auto text-[10px]">
              {indianStocks.slice(0, 50).map((s) => (
                <button key={s} className={`w-full text-left px-2 py-1 rounded glass-hover ${s === stock ? "text-[#3B82F6]" : "text-[#94A3B8]"}`}
                  onClick={() => { setStock(s); setShowDropdown(false); }}
                >{s}</button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 flex min-h-0">
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 min-h-0 relative">
            <div ref={chartRef} className="w-full h-full" />
            {candleData.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center text-[#4B5563] text-xs">Loading {stock}...</div>
            )}
          </div>
        </div>

        <div className="w-64 shrink-0 border-l border-white/5 overflow-y-auto p-3 space-y-3 text-[11px]">
          <div>
            <div className="text-[10px] text-[#4B5563] font-medium mb-1 uppercase tracking-wider">Parameters</div>
            {paramsList.map((p) => (
              <div key={p.key} className="flex items-center justify-between mb-1.5">
                <span className="text-[#94A3B8]">{p.label}</span>
                <div className="flex items-center gap-1">
                  <button className="btn-ghost p-0.5 text-[#4B5563]" onClick={() => setParams((s) => ({ ...s, [p.key]: Math.max(p.min, (s as any)[p.key] - p.step) }))}>−</button>
                  <span className="text-white tabular-nums w-8 text-right">{(params as any)[p.key]}</span>
                  <button className="btn-ghost p-0.5 text-[#4B5563]" onClick={() => setParams((s) => ({ ...s, [p.key]: Math.min(p.max, (s as any)[p.key] + p.step) }))}>+</button>
                </div>
              </div>
            ))}
          </div>

          <div className="border-t border-white/5 pt-2">
            <div className="text-[10px] text-[#4B5563] font-medium mb-1 uppercase tracking-wider">
              Detection {detectedPattern && <span className={`ml-1 ${detectedPattern.isTrigger ? "text-[#22C55E]" : "text-[#F59E0B]"}`}>{detectedPattern.isTrigger ? "MATCH" : "PARTIAL"}</span>}
            </div>
            {detectedPattern ? (
              <div className="space-y-1.5">
                <div className="flex justify-between"><span className="text-[#4B5563]">Date</span><span className="text-white">{detectedPattern.triggerDate}</span></div>
                <div className="flex justify-between"><span className="text-[#4B5563]">Type</span><span className={detectedPattern.triggerType === "BULLISH" ? "text-[#22C55E]" : "text-[#EF4444]"}>{detectedPattern.triggerType}</span></div>
                <div className="flex justify-between"><span className="text-[#4B5563]">Body</span><span className="text-[#8B5CF6]">{detectedPattern.bodyVsAvg}x</span></div>
                <div className="flex justify-between"><span className="text-[#4B5563]">Volume</span><span className="text-[#3B82F6]">{detectedPattern.volVsAvg}x</span></div>
                <div className="flex justify-between"><span className="text-[#4B5563]">RSI</span><span className="text-[#F59E0B]">{detectedPattern.rsi}</span></div>
                <div className="flex justify-between"><span className="text-[#4B5563]">Consol</span><span className="text-[#22C55E]">{detectedPattern.consolCount}d</span></div>
                <div className="flex justify-between"><span className="text-[#4B5563]">Δ%</span><span className={detectedPattern.changePct >= 0 ? "text-[#22C55E]" : "text-[#EF4444]"}>{detectedPattern.changePct > 0 ? "+" : ""}{detectedPattern.changePct}%</span></div>
              </div>
            ) : (
              <div className="text-[#4B5563] text-[10px]">Click a candle on the chart</div>
            )}
          </div>

          <div className="border-t border-white/5 pt-2">
            <div className="text-[10px] text-[#4B5563] font-medium mb-1 uppercase tracking-wider">Save Pattern</div>
            <div className="flex gap-1 mb-1">
              <input className="input flex-1 text-[10px] px-2 py-1" placeholder="Pattern name"
                value={name} onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && savePattern()} />
            </div>
            <div className="flex gap-1">
              <select className="input text-[10px] px-2 py-1 flex-1" value={category} onChange={(e) => setCategory(e.target.value)}>
                <option>BCC Variant</option>
                <option>Squeeze Variant</option>
                <option>Custom</option>
              </select>
              <button onClick={savePattern} disabled={!name.trim() || !detectedPattern} className="btn-ghost p-1 text-[#22C55E] disabled:opacity-30">
                <Save size={12} />
              </button>
            </div>
          </div>

          {savedPatterns.length > 0 && (
            <div className="border-t border-white/5 pt-2">
              <div className="text-[10px] text-[#4B5563] font-medium mb-1 uppercase tracking-wider">Saved ({savedPatterns.length})</div>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {savedPatterns.toReversed().map((p) => (
                  <div key={p.id} className="flex items-center justify-between glass rounded px-2 py-1">
                    <div>
                      <div className="text-[10px] text-white">{p.name}</div>
                      <div className="text-[8px] text-[#4B5563]">{p.stock} · {p.category} · {p.bodyVsAvg}x/{p.volVsAvg}x</div>
                    </div>
                    <button onClick={() => deletePattern(p.id)} className="btn-ghost p-0.5 text-[#EF4444]"><Trash2 size={9} /></button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
