"use client";

import { useStore } from "@/store/useStore";
import { Sparkles, TrendingUp, TrendingDown, Target } from "lucide-react";
import { useEffect } from "react";

export default function AIInsights() {
  const { aiInsight, focusPattern, selectedStock, fetchAIInsight } = useStore();

  useEffect(() => {
    if (focusPattern) {
      fetchAIInsight(selectedStock);
    }
  }, [focusPattern, selectedStock, fetchAIInsight]);

  const insight = aiInsight || {
    pattern: "Big Candle + Consolidation",
    reason: "Click a stock in the scanner to see AI insights for its pattern.",
    accuracy: 0, winRate: 0, risk: 0, reward: 0, trend: "Neutral",
  };

  const MetricCard = ({ label, value, color }: { label: string; value: string; color?: string }) => (
    <div className="glass rounded-lg px-3 py-2">
      <div className="text-[10px] text-[#4B5563] mb-0.5">{label}</div>
      <div className={`text-sm font-semibold ${color || "text-white"}`}>{value}</div>
    </div>
  );

  return (
    <div className="panel h-full flex flex-col">
      <div className="panel-header shrink-0 drag-handle cursor-grab active:cursor-grabbing">
        <div className="flex items-center gap-2">
          <Sparkles size={12} className="text-[#3B82F6]" />
          <span className="text-xs font-semibold">AI INSIGHTS</span>
        </div>
        <div className="flex items-center gap-2 text-[10px]">
          <span className="tag-ai">Live</span>
          {focusPattern && <span className="text-[#4B5563]">{selectedStock}</span>}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        <div className="flex items-center justify-between">
          <div className="tag tag-pattern text-xs">{insight.pattern}</div>
          <div className="flex items-center gap-1 text-xs">
            {insight.trend === "Bullish" ? (
              <TrendingUp size={12} className="text-[#22C55E]" />
            ) : insight.trend === "Bearish" ? (
              <TrendingDown size={12} className="text-[#EF4444]" />
            ) : null}
            <span className={insight.trend === "Bullish" ? "text-[#22C55E]" : insight.trend === "Bearish" ? "text-[#EF4444]" : "text-[#94A3B8]"}>
              {insight.trend}
            </span>
          </div>
        </div>

        <div>
          <div className="flex justify-between text-[10px] mb-1">
            <span className="text-[#4B5563]">AI Confidence</span>
            <span className="text-[#3B82F6] font-medium">{insight.accuracy.toFixed(0)}%</span>
          </div>
          <div className="h-2 rounded-full bg-white/5 overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-[#3B82F6] to-[#8B5CF6]"
              style={{ width: `${insight.accuracy}%` }}
            />
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2">
          <MetricCard label="Win Rate" value={insight.winRate > 0 ? `${insight.winRate.toFixed(0)}%` : "—"} color="text-[#22C55E]" />
          <MetricCard label="Risk" value={insight.risk > 0 ? `${insight.risk.toFixed(1)}%` : "—"} color="text-[#EF4444]" />
          <MetricCard label="Reward" value={insight.reward > 0 ? `${insight.reward.toFixed(1)}%` : "—"} color="text-[#22C55E]" />
        </div>

        {insight.risk > 0 && (
          <div className="flex items-center justify-between glass rounded-lg px-3 py-2">
            <div className="flex items-center gap-2">
              <Target size={12} className="text-[#F59E0B]" />
              <span className="text-[10px] text-[#4B5563]">Risk:Reward</span>
            </div>
            <span className="text-sm font-semibold text-[#F59E0B]">
              1:{((insight.reward / insight.risk) || 2).toFixed(1)}
            </span>
          </div>
        )}

        <div>
          <div className="text-[10px] text-[#4B5563] mb-1.5 uppercase tracking-wider">Analysis</div>
          <div className="text-xs text-[#94A3B8] leading-relaxed bg-white/[0.02] rounded-lg p-3">
            {insight.reason}
          </div>
        </div>

        {focusPattern && (
          <div>
            <div className="text-[10px] text-[#4B5563] mb-1.5 uppercase tracking-wider">Pattern Details</div>
            <div className="flex flex-wrap gap-1.5">
              <span className="text-[10px] px-2 py-0.5 rounded-md bg-[#8B5CF6]/10 text-[#8B5CF6]">
                Body: {focusPattern.bodyVsAvg.toFixed(1)}x
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-md bg-[#3B82F6]/10 text-[#3B82F6]">
                Vol: {focusPattern.volVsAvg.toFixed(1)}x
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-md bg-[#F59E0B]/10 text-[#F59E0B]">
                RSI: {focusPattern.rsi}
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-md bg-[#22C55E]/10 text-[#22C55E]">
                Consol: {focusPattern.consolCount}d
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-md bg-[#EF4444]/10 text-[#EF4444]">
                Status: {focusPattern.status}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
