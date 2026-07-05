import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const DATA_DIR = "C:/Users/pc/Downloads/stock hist data/comprehensive_data";
const CACHE_DIR = "C:/Users/pc/Downloads/stock hist data/ai-pattern-screener/daily_cache";

const LOOKBACK = 20;
const CONSOL_MAX_PCT = 6;
const BREAKOUT_MIN_PCT = 0.3;

function loadCachedCandles(symbol: string): any[] | null {
  const cachePath = path.join(CACHE_DIR, `${symbol}.json`);
  if (!fs.existsSync(cachePath)) return null;
  try {
    const cached = JSON.parse(fs.readFileSync(cachePath, "utf-8"));
    if (cached.candles && cached.candles.length >= 22) return cached.candles;
  } catch {}
  return null;
}

function parseTail(raw: string, tail = 60): any[] {
  const lines = raw.trim().split("\n");
  if (lines.length < 3) return [];
  const header = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const start = Math.max(1, lines.length - tail);
  const result: any[] = [];
  for (let i = start; i < lines.length; i++) {
    const vals = lines[i].split(",");
    const row: any = {};
    for (let j = 0; j < header.length; j++) row[header[j]] = vals[j]?.trim() || "";
    result.push(row);
  }
  return result;
}

export async function GET(req: NextRequest) {
  try {
    const files = fs.readdirSync(DATA_DIR).filter((f) => f.endsWith("_ONE_DAY.csv"));
    const results: any[] = [];

    for (const file of files) {
      const sym = file.replace("_ONE_DAY.csv", "");

      try {
        const cachedCandles = loadCachedCandles(sym);
        let closes: number[], highs: number[], lows: number[], volumes: number[], price: number;

        if (cachedCandles) {
          const use = cachedCandles.slice(-Math.max(LOOKBACK + 5, 30));
          closes = use.map((c: any) => c.close);
          highs = use.map((c: any) => c.high);
          lows = use.map((c: any) => c.low);
          volumes = use.map((c: any) => c.volume || 0);
          price = closes[closes.length - 1];
        } else {
          const raw = fs.readFileSync(path.join(DATA_DIR, file), "utf-8");
          const records = parseTail(raw, LOOKBACK + 5);
          if (records.length < LOOKBACK) continue;
          closes = records.map((r: any) => parseFloat(r.close) || 0);
          highs = records.map((r: any) => parseFloat(r.high) || 0);
          lows = records.map((r: any) => parseFloat(r.low) || 0);
          volumes = records.map((r: any) => parseInt(r.volume) || 0);
          price = closes[closes.length - 1];
        }

        if (closes.length < LOOKBACK) continue;

        // Use last LOOKBACK bars to define the range
        const recentHigh = Math.max(...highs.slice(-LOOKBACK));
        const recentLow = Math.min(...lows.slice(-LOOKBACK));
        const rangePct = recentLow > 0 ? ((recentHigh - recentLow) / recentLow) * 100 : 0;

        // Calculate close vs range bounds
        const aboveHighPct = ((price - recentHigh) / recentHigh) * 100;
        const belowLowPct = ((recentLow - price) / recentLow) * 100;

        let type: string;
        let breakoutPct = 0;

        if (aboveHighPct >= BREAKOUT_MIN_PCT) {
          type = "BULLISH BREAKOUT";
          breakoutPct = aboveHighPct;
        } else if (belowLowPct >= BREAKOUT_MIN_PCT) {
          type = "BEARISH BREAKOUT";
          breakoutPct = belowLowPct;
        } else if (rangePct <= CONSOL_MAX_PCT) {
          type = "CONSOLIDATING";
        } else {
          continue;
        }

        // Recent slope for direction/tightness
        const recentCloses = closes.slice(-5);
        const slope = recentCloses.length >= 2 ? recentCloses[recentCloses.length - 1] - recentCloses[0] : 0;

        // Volume confirmation
        const avgVol = volumes.slice(-LOOKBACK).reduce((a, b) => a + b, 0) / LOOKBACK;
        const lastVol = volumes[volumes.length - 1] || 0;
        const volRatio = avgVol > 0 ? lastVol / avgVol : 0;

        // Strength score
        let strength = 0;
        if (type === "CONSOLIDATING") {
          strength = Math.round((CONSOL_MAX_PCT - rangePct) * 10);
        } else {
          strength = Math.round(breakoutPct * 10 + (volRatio > 1.5 ? 10 : 0));
        }

        results.push({
          ticker: sym,
          price: Math.round(price * 100) / 100,
          rangeHigh: Math.round(recentHigh * 100) / 100,
          rangeLow: Math.round(recentLow * 100) / 100,
          rangePct: Math.round(rangePct * 100) / 100,
          type,
          breakoutPct: Math.round(breakoutPct * 100) / 100,
          slope: Math.round(slope * 100) / 100,
          strength,
          volRatio: Math.round(volRatio * 100) / 100,
        });
      } catch {
        continue;
      }
    }

    // Sort: breakouts first (by strength), then consolidating (by tightest range)
    results.sort((a, b) => {
      const aScore = a.type === "BULLISH BREAKOUT" ? 3 : a.type === "BEARISH BREAKOUT" ? 2 : 1;
      const bScore = b.type === "BULLISH BREAKOUT" ? 3 : b.type === "BEARISH BREAKOUT" ? 2 : 1;
      if (aScore !== bScore) return bScore - aScore;
      return b.strength - a.strength;
    });
    results.forEach((r, i) => (r.rank = i + 1));

    return NextResponse.json({ stocks: results, total: results.length, generatedAt: new Date().toISOString() });
  } catch (err) {
    console.error("Consolidation breakout scanner error:", err);
    return NextResponse.json({ error: "Consolidation breakout scanner failed" }, { status: 500 });
  }
}
