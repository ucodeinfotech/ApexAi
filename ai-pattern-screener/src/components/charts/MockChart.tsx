"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, ColorType, CrosshairMode, LineStyle } from "lightweight-charts";
import { useStore } from "@/store/useStore";
import { Maximize2, Camera, RotateCcw, Crosshair, Minus, Plus, Loader2 } from "lucide-react";

export default function MockChart() {
  const chartRef = useRef<HTMLDivElement>(null);
  const { selectedStock, selectedTimeframe } = useStore();
  const [candleData, setCandleData] = useState<any[]>([]);
  const [demaAtr, setDemaAtr] = useState<number[]>([]);
  const [loading, setLoading] = useState(false);
  const [price, setPrice] = useState<string>("—");
  const [change, setChange] = useState<string>("");
  const [changeColor, setChangeColor] = useState<string>("#94A3B8");

  useEffect(() => {
    if (!selectedStock) return;
    const fetchData = async () => {
      setLoading(true);
      try {
        const res = await fetch(`/api/stock/${selectedStock}`);
        if (!res.ok) throw new Error("Not found");
        const data = await res.json();
        const sorted = (data.candles || []).sort((a: any, b: any) => a.time - b.time);
          const chartData = sorted.map((c: any) => ({ ...c, time: new Date(c.time * 1000).toISOString().split("T")[0] }));
          setCandleData(chartData);
          setDemaAtr(data.demaAtr?.slice() || []);
          if (sorted.length > 0) {
          const last = sorted[sorted.length - 1];
          const prev = sorted.length > 1 ? sorted[sorted.length - 2] : last;
          const chg = ((last.close - prev.close) / prev.close * 100);
          setPrice(last.close.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
          setChange(`${chg >= 0 ? "+" : ""}${chg.toFixed(2)}%`);
          setChangeColor(chg >= 0 ? "#22C55E" : "#EF4444");
        }
      } catch {
        setCandleData([]);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [selectedStock]);

  useEffect(() => {
    if (!chartRef.current || candleData.length === 0) return;

    const chart = createChart(chartRef.current, {
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#4B5563", fontSize: 11 },
      grid: { vertLines: { color: "rgba(255,255,255,0.03)", style: LineStyle.Dashed }, horzLines: { color: "rgba(255,255,255,0.03)", style: LineStyle.Dashed } },
      crosshair: { mode: CrosshairMode.Normal, vertLine: { color: "rgba(255,255,255,0.2)", width: 1, style: LineStyle.Dashed, labelBackgroundColor: "#3B82F6" }, horzLine: { color: "rgba(255,255,255,0.2)", width: 1, style: LineStyle.Dashed, labelBackgroundColor: "#3B82F6" } },
      rightPriceScale: { borderColor: "rgba(255,255,255,0.08)", scaleMargins: { top: 0.05, bottom: 0.25 } },
      timeScale: { borderColor: "rgba(255,255,255,0.08)", timeVisible: true, secondsVisible: false },
      handleScroll: true, handleScale: true,
    });

    const candleSeries = chart.addCandlestickSeries({ upColor: "#22C55E", downColor: "#EF4444", borderDownColor: "#EF4444", borderUpColor: "#22C55E", wickDownColor: "#EF4444", wickUpColor: "#22C55E" });
    candleSeries.setData(candleData);

    // DEMA ATR line
    if (demaAtr.length === candleData.length) {
      const demaLine = chart.addLineSeries({ color: "#8B5CF6", lineWidth: 2, priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: true });
      demaLine.setData(candleData.map((d: any, i: number) => ({ time: d.time, value: demaAtr[i] })));
    }

    const volumeSeries = chart.addHistogramSeries({ priceFormat: { type: "volume" }, priceScaleId: "volume" });
    chart.priceScale("volume").applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
    volumeSeries.setData(candleData.map((d: any) => ({ time: d.time, value: d.volume, color: d.close >= d.open ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)" })));

    chart.timeScale().fitContent();

    const resize = () => { if (chartRef.current) chart.applyOptions({ width: chartRef.current.clientWidth, height: chartRef.current.clientHeight }); };
    window.addEventListener("resize", resize);
    setTimeout(resize, 100);
    return () => { window.removeEventListener("resize", resize); chart.remove(); };
  }, [candleData, selectedTimeframe]);

  return (
    <div className="panel h-full flex flex-col">
      <div className="panel-header shrink-0 drag-handle cursor-grab active:cursor-grabbing">
        <div className="flex items-center gap-3">
          <div><span className="text-sm font-semibold">{selectedStock}</span><span className="text-xs text-[#94A3B8] ml-2">{selectedTimeframe}</span></div>
          <div className="flex items-center gap-2 text-xs">
            {loading ? <Loader2 size={10} className="animate-spin text-[#3B82F6]" /> : (
              <><span className="text-white font-medium tabular-nums">{price}</span><span className="tabular-nums" style={{ color: changeColor }}>{change}</span></>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1">
          <button className="btn-ghost p-1"><Crosshair size={11} /></button>
          <button className="btn-ghost p-1"><Plus size={11} /></button>
          <button className="btn-ghost p-1"><Minus size={11} /></button>
          <button className="btn-ghost p-1"><RotateCcw size={11} /></button>
          <div className="w-px h-4 mx-0.5" style={{ background: "rgba(255,255,255,0.08)" }} />
          <button className="btn-ghost p-1"><Camera size={11} /></button>
          <button className="btn-ghost p-1"><Maximize2 size={11} /></button>
        </div>
      </div>
      {candleData.length === 0 && !loading && (
        <div className="flex-1 flex items-center justify-center text-[#4B5563] text-xs">No data</div>
      )}
      <div ref={chartRef} className="flex-1 min-h-0" />
    </div>
  );
}
