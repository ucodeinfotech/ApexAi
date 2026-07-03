import { NextResponse } from "next/server";
import fs from "fs";

const TICKS_FILE = "C:/Users/pc/Downloads/stock hist data/ai-pattern-screener/ticks.jsonl";
const TOKENS_FILE = "C:/Users/pc/Downloads/stock hist data/nse_tokens.json";

export async function GET() {
  try {
    if (!fs.existsSync(TICKS_FILE)) {
      return NextResponse.json({ active: false, tickCount: 0, uniqueStocks: 0, totalVolume: 0, timestamp: new Date().toISOString(), topGainers: [], topLosers: [], summary: { advancers: 0, decliners: 0, unchanged: 0, totalVolume: 0 } });
    }

    const raw = fs.readFileSync(TICKS_FILE, "utf-8");
    const lines = raw.trim().split("\n").filter(Boolean);
    const ticks = lines.slice(-200).map((l) => JSON.parse(l));

    // Reverse token map
    const tokenMap: Record<string, string> = {};
    if (fs.existsSync(TOKENS_FILE)) {
      const tokens = JSON.parse(fs.readFileSync(TOKENS_FILE, "utf-8"));
      for (const [sym, tok] of Object.entries(tokens)) {
        tokenMap[tok as string] = sym;
      }
    }

    // Deduplicate by token, keeping latest
    const latest = new Map<string, any>();
    for (const t of ticks) {
      latest.set(t.token, { ...t, symbol: tokenMap[t.token] || t.token });
    }

    const allTicks = Array.from(latest.values());
    const active = allTicks.filter((t) => t.ltp > 0);
    const withChange = active.filter((t) => t.changePct !== 0);

    const topGainers = [...withChange].sort((a, b) => b.changePct - a.changePct).slice(0, 10);
    const topLosers = [...withChange].sort((a, b) => a.changePct - b.changePct).slice(0, 10);

    const totalVolume = active.reduce((s, t) => s + (t.volume || 0), 0);

    return NextResponse.json({
      active: true,
      tickCount: lines.length,
      uniqueStocks: active.length,
      totalVolume,
      timestamp: new Date().toISOString(),
      topGainers: topGainers.map((t) => ({ symbol: t.symbol, ltp: t.ltp, changePct: t.changePct, volume: t.volume })),
      topLosers: topLosers.map((t) => ({ symbol: t.symbol, ltp: t.ltp, changePct: t.changePct, volume: t.volume })),
      summary: {
        advancers: active.filter((t) => t.changePct > 0).length,
        decliners: active.filter((t) => t.changePct < 0).length,
        unchanged: active.filter((t) => t.changePct === 0).length,
        totalVolume,
      },
    });
  } catch {
    return NextResponse.json({ active: false, ticks: [], topMovers: [], summary: { total: 0 } });
  }
}
