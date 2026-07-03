"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, ColorType, CrosshairMode, LineStyle, type IChartApi, type CandlestickSeriesPartialOptions } from "lightweight-charts";
import { useStore } from "@/store/useStore";
import { Maximize2, Camera, RotateCcw, Crosshair, Minus, Plus, X } from "lucide-react";

export default function CandlestickChart() {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<IChartApi | null>(null);
  const { selectedStock, selectedTimeframe, focusPattern, setFocusPattern, showChartModal, setShowChartModal } = useStore();
  const [candleData, setCandleData] = useState<any[]>([]);
  const [allPatterns, setAllPatterns] = useState<any[]>([]);
  const [demaAtr, setDemaAtr] = useState<number[]>([]);
  const [loading, setLoading] = useState(false);

  // Fetch candle data when stock or focusPattern changes
  useEffect(() => {
    if (!showChartModal || !selectedStock) return;

    const fetchData = async () => {
      setLoading(true);
      try {
        const res = await fetch(`/api/stock/${selectedStock}`);
        if (!res.ok) throw new Error("Not found");
        const data = await res.json();
        const sorted = (data.candles || []).sort((a: any, b: any) => a.time - b.time);
        setCandleData(sorted);
        setAllPatterns(data.patterns || []);
        setDemaAtr(data.demaAtr?.slice() || []);
      } catch {
        setCandleData([]);
        setAllPatterns([]);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [selectedStock, showChartModal]);

  // Render chart when candle data is available
  useEffect(() => {
    if (!chartRef.current || candleData.length === 0) return;

    if (chartInstance.current) {
      chartInstance.current.remove();
      chartInstance.current = null;
    }

    const chart = createChart(chartRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#4B5563",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.03)", style: LineStyle.Dashed },
        horzLines: { color: "rgba(255,255,255,0.03)", style: LineStyle.Dashed },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "rgba(255,255,255,0.2)", width: 1, style: LineStyle.Dashed, labelBackgroundColor: "#3B82F6" },
        horzLine: { color: "rgba(255,255,255,0.2)", width: 1, style: LineStyle.Dashed, labelBackgroundColor: "#3B82F6" },
      },
      rightPriceScale: {
        borderColor: "rgba(255,255,255,0.08)",
        scaleMargins: { top: 0.05, bottom: 0.25 },
      },
      timeScale: {
        borderColor: "rgba(255,255,255,0.08)",
        timeVisible: true,
        secondsVisible: false,
      },
      handleScroll: true,
      handleScale: true,
    });
    chartInstance.current = chart;

    // Candles
    const candleSeries = chart.addCandlestickSeries({
      upColor: "#22C55E",
      downColor: "#EF4444",
      borderDownColor: "#EF4444",
      borderUpColor: "#22C55E",
      wickDownColor: "#EF4444",
      wickUpColor: "#22C55E",
    });

    candleSeries.setData(candleData);

    // DEMA ATR line
    if (demaAtr.length === candleData.length) {
      const demaLine = chart.addLineSeries({ color: "#8B5CF6", lineWidth: 2, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: true });
      demaLine.setData(candleData.map((d: any, i: number) => ({ time: d.time, value: demaAtr[i] })));
    }

    // Volume
    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });
    chart.priceScale("volume").applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
    volumeSeries.setData(
      candleData.map((d: any) => ({
        time: d.time,
        value: d.volume,
        color: d.close >= d.open ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)",
      }))
    );

    // ─── RENDER CONSOLIDATION ZONE FOR ALL PATTERNS ───
    const allChartMarkers: any[] = [];
    const maxRecent = Math.min(allPatterns.length, 20);
    const recentPatterns = allPatterns.slice(-maxRecent);

    recentPatterns.forEach((p: any) => {
      const isFocus = focusPattern && p.triggerTime === focusPattern.triggerTime;
      const isBullish = p.triggerType === "BULLISH";
      const tIdx = candleData.findIndex((d: any) => d.time === p.triggerTime);
      const eIdx = p.consolEndTime ? candleData.findIndex((d: any) => d.time === p.consolEndTime) : -1;

      // Trigger marker
      if (isFocus) {
        allChartMarkers.push({
          time: p.triggerTime,
          position: isBullish ? "belowBar" as const : "aboveBar" as const,
          color: isBullish ? "#22C55E" : "#EF4444",
          shape: isBullish ? "arrowUp" as const : "arrowDown" as const,
          text: `BCC ${p.triggerType}`,
          size: 1.5,
        });
      } else {
        allChartMarkers.push({
          time: p.triggerTime,
          position: isBullish ? "belowBar" as const : "aboveBar" as const,
          color: isBullish ? "rgba(34,197,94,0.5)" : "rgba(239,68,68,0.5)",
          shape: isBullish ? "arrowUp" as const : "arrowDown" as const,
          text: "",
          size: 0.6,
        });
      }

      // Consolidation zone
      if (tIdx >= 0 && eIdx > tIdx) {
        const opacity = isFocus ? 0.25 : 0.1;
        const lineWidth = isFocus ? 2 : 1;
        const lineColor = isBullish ? `rgba(34,197,94,${isFocus ? 0.6 : 0.3})` : `rgba(239,68,68,${isFocus ? 0.6 : 0.3})`;
        const fillColor = isBullish ? `rgba(34,197,94,${opacity})` : `rgba(239,68,68,${opacity})`;
        const consolHigh = p.consolHigh;
        const consolLow = p.consolLow;
        const lStart = Math.max(0, tIdx - 1);
        const lEnd = Math.min(candleData.length - 1, eIdx + 2);

        // High line
        const hl = chart.addLineSeries({
          color: lineColor, lineWidth, lineStyle: LineStyle.Dashed,
          priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
        });
        const hd: any[] = [];
        for (let k = lStart; k <= lEnd; k++) hd.push({ time: candleData[k].time, value: consolHigh });
        if (hd.length >= 2) hl.setData(hd);

        // Low line
        const ll = chart.addLineSeries({
          color: lineColor, lineWidth, lineStyle: LineStyle.Dashed,
          priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
        });
        const ld: any[] = [];
        for (let k = lStart; k <= lEnd; k++) ld.push({ time: candleData[k].time, value: consolLow });
        if (ld.length >= 2) ll.setData(ld);

        // Area fill
        const ar = chart.addAreaSeries({
          lineColor: "transparent",
          topColor: fillColor,
          bottomColor: fillColor,
          priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
        });
        const ad: any[] = [];
        for (let k = lStart; k <= lEnd; k++) ad.push({ time: candleData[k].time, value: consolLow + (consolHigh - consolLow) / 2 });
        if (ad.length >= 2) ar.setData(ad);

        // Consolidation dots
        for (let k = tIdx + 1; k <= eIdx && k < candleData.length; k++) {
          allChartMarkers.push({
            time: candleData[k].time,
            position: "belowBar" as const,
            color: isFocus ? "#8B5CF6" : "rgba(139,92,246,0.3)",
            shape: "circle" as const,
            size: isFocus ? 0.8 : 0.4,
            text: "",
          });
        }

        // Consolidation label for focus pattern
        if (isFocus) {
          const midIdx = Math.floor((tIdx + 1 + eIdx) / 2);
          allChartMarkers.push({
            time: candleData[midIdx]?.time || candleData[eIdx].time,
            position: "aboveBar" as const,
            color: "#8B5CF6",
            shape: "square" as const,
            size: 0.8,
            text: " CONSOLIDATION ",
          });
        }
      }
    });

    allChartMarkers.sort((a: any, b: any) => a.time - b.time);
    candleSeries.setMarkers(allChartMarkers.slice(0, 300));

    if (focusPattern) {
      chart.timeScale().scrollToPosition(0, false);
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartRef.current) {
        chart.applyOptions({ width: chartRef.current.clientWidth, height: chartRef.current.clientHeight });
      }
    };
    window.addEventListener("resize", handleResize);
    setTimeout(handleResize, 100);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartInstance.current = null;
    };
  }, [candleData, focusPattern, allPatterns, demaAtr]);

  if (!showChartModal) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => { setShowChartModal(false); setFocusPattern(null); }}>
      <div
        className="panel w-[90vw] h-[85vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="panel-header shrink-0">
          <div className="flex items-center gap-3">
            <div>
              <span className="text-sm font-semibold">{selectedStock}</span>
              <span className="text-xs text-[#94A3B8] ml-2">{selectedTimeframe}</span>
            </div>
            {focusPattern && (
              <div className="flex items-center gap-2 text-[10px]">
                <span className={`tag ${focusPattern.triggerType === "BULLISH" ? "tag-bullish" : "tag-bearish"}`}>
                  {focusPattern.triggerType}
                </span>
                <span className="text-[#4B5563]">|</span>
                <span className="text-[#8B5CF6]">BCC Pattern</span>
                <span className="text-[#4B5563]">|</span>
                <span className="text-[#22C55E]">{focusPattern.bodyVsAvg.toFixed(1)}x body</span>
                <span className="text-[#3B82F6]">{focusPattern.volVsAvg.toFixed(1)}x vol</span>
                <span className="text-[#F59E0B]">RSI {focusPattern.rsi}</span>
                <span className="text-[#4B5563]">|</span>
                <span className="text-[#94A3B8]">Consol: {focusPattern.consolCount}d ({focusPattern.consolRangePct.toFixed(1)}%)</span>
                <span className={`tag ${focusPattern.status === "CONSOLIDATING" ? "tag-warning" : focusPattern.status === "BROKEN UP" ? "tag-bullish" : "tag-bearish"}`}>
                  {focusPattern.status}
                </span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            {loading && <span className="text-[10px] text-[#3B82F6]">Loading...</span>}
            <div className="flex items-center gap-1">
              <button className="btn-ghost p-1"><Crosshair size={11} /></button>
              <button className="btn-ghost p-1"><Plus size={11} /></button>
              <button className="btn-ghost p-1"><Minus size={11} /></button>
              <button className="btn-ghost p-1"><RotateCcw size={11} /></button>
              <button className="btn-ghost p-1"><Camera size={11} /></button>
              <button className="btn-ghost p-1"><Maximize2 size={11} /></button>
              <div className="w-px h-4 mx-1" style={{ background: "rgba(255,255,255,0.08)" }} />
              <button
                onClick={() => { setShowChartModal(false); setFocusPattern(null); }}
                className="btn-ghost p-1 text-[#EF4444] hover:text-[#EF4444]"
              >
                <X size={14} />
              </button>
            </div>
          </div>
        </div>
        <div className="flex-1 min-h-0 relative">
          {candleData.length === 0 && !loading && (
            <div className="absolute inset-0 flex items-center justify-center text-[#4B5563] text-sm">
              No data for {selectedStock}
            </div>
          )}
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center text-[#3B82F6] text-sm">
              Loading {selectedStock}...
            </div>
          )}
          <div ref={chartRef} className="w-full h-full" />
        </div>
      </div>
    </div>
  );
}
