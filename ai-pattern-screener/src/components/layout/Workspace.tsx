"use client";

import dynamic from "next/dynamic";
import { Responsive, WidthProvider } from "react-grid-layout";
import "react-grid-layout/css/styles.css";
import MockChart from "@/components/charts/MockChart";
import ScannerTable from "@/components/scanner/ScannerTable";
import AIInsights from "@/components/insights/AIInsights";
import PatternComparison from "@/components/insights/PatternComparison";
import BreadthCards from "@/components/market/BreadthCards";
import PatternBuilder from "@/components/patterns/PatternBuilder";
import DemaAtrPanel from "@/components/scanner/DemaAtrPanel";
import ConsolidationBreakoutPanel from "@/components/scanner/ConsolidationBreakoutPanel";
import { useStore } from "@/store/useStore";
import { useState, useEffect } from "react";

const ResponsiveGridLayout = WidthProvider(Responsive);

const defaultLayouts = {
  lg: [
    { i: "chart", x: 0, y: 0, w: 6, h: 7, minW: 4, minH: 4 },
    { i: "insights", x: 6, y: 0, w: 3, h: 7, minW: 3, minH: 4 },
    { i: "breadth", x: 9, y: 0, w: 3, h: 7, minW: 3, minH: 6 },
    { i: "scanner", x: 0, y: 7, w: 7, h: 7, minW: 4, minH: 4 },
    { i: "dema-atr", x: 7, y: 7, w: 3, h: 7, minW: 3, minH: 4 },
    { i: "consolidation-breakout", x: 10, y: 7, w: 2, h: 7, minW: 2, minH: 4 },
    { i: "comparison", x: 0, y: 14, w: 12, h: 7, minW: 3, minH: 4 },
    { i: "builder", x: 0, y: 14, w: 12, h: 8, minW: 6, minH: 6 },
  ],
  md: [
    { i: "chart", x: 0, y: 0, w: 6, h: 7, minW: 4, minH: 4 },
    { i: "insights", x: 6, y: 0, w: 6, h: 5, minW: 3, minH: 4 },
    { i: "breadth", x: 0, y: 7, w: 12, h: 5, minW: 3, minH: 6 },
    { i: "scanner", x: 0, y: 12, w: 12, h: 7, minW: 4, minH: 4 },
    { i: "dema-atr", x: 0, y: 19, w: 12, h: 5, minW: 3, minH: 4 },
    { i: "consolidation-breakout", x: 0, y: 24, w: 12, h: 5, minW: 2, minH: 4 },
    { i: "comparison", x: 0, y: 29, w: 12, h: 5, minW: 3, minH: 4 },
    { i: "builder", x: 0, y: 24, w: 12, h: 8, minW: 6, minH: 6 },
  ],
};

const panelComponents: Record<string, React.ReactNode> = {
  chart: <MockChart />,
  insights: <AIInsights />,
  scanner: <ScannerTable />,
  "dema-atr": <DemaAtrPanel />,
  "consolidation-breakout": <ConsolidationBreakoutPanel />,
  comparison: <PatternComparison />,
  breadth: <BreadthCards />,
  builder: <PatternBuilder />,
};

const panelTitles: Record<string, string> = {
  chart: "Chart",
  insights: "AI Insights",
  scanner: "Scanner",
  "dema-atr": "DEMA ATR",
  "consolidation-breakout": "Consol / Breakout",
  comparison: "Pattern Comparison",
  breadth: "Market Breadth",
  builder: "Pattern Builder",
};

export default function Workspace() {
  const [mounted, setMounted] = useState(false);
  const { panelOrder, panelVisibility, togglePanel, fetchBreadth } = useStore();

  useEffect(() => {
    setMounted(true);
    fetchBreadth();
  }, [fetchBreadth]);

  if (!mounted) {
    return (
      <div className="flex-1 p-3 grid grid-cols-12 grid-rows-[7fr_7fr] gap-3 min-h-0" style={{ height: "calc(100vh - 64px - 32px)" }}>
        <div className="col-span-6 row-span-1"><MockChart /></div>
        <div className="col-span-3 row-span-1"><AIInsights /></div>
        <div className="col-span-3 row-span-1"><BreadthCards /></div>
        <div className="col-span-7 row-span-1"><ScannerTable /></div>
        <div className="col-span-3 row-span-1"><DemaAtrPanel /></div>
        <div className="col-span-2 row-span-1"><ConsolidationBreakoutPanel /></div>
      </div>
    );
  }

  const visibleItems = panelOrder.filter((id) => panelVisibility[id]);

  const layouts = {
    lg: defaultLayouts.lg.filter((l) => visibleItems.includes(l.i)),
    md: defaultLayouts.md.filter((l) => visibleItems.includes(l.i)),
  };

  return (
    <div className="flex-1 p-3 min-h-0" style={{ height: "calc(100vh - 64px - 32px)" }}>
      <ResponsiveGridLayout
        className="layout"
        layouts={layouts}
        breakpoints={{ lg: 1400, md: 1024 }}
        cols={{ lg: 12, md: 12 }}
        rowHeight={36}
        isResizable
        isDraggable
        compactType="vertical"
        margin={[12, 12]}
        draggableHandle=".drag-handle"
      >
        {visibleItems.map((id) => (
          <div key={id} className="panel overflow-hidden drag-container" style={{ background: "rgba(255,255,255,0.03)", backdropFilter: "blur(12px)" }}>
            {panelComponents[id]}
          </div>
        ))}
      </ResponsiveGridLayout>
    </div>
  );
}
