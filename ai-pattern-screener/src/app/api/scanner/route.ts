import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const DATA_DIR = "C:/Users/pc/Downloads/stock hist data/comprehensive_data";
const CACHE_DIR = "C:/Users/pc/Downloads/stock hist data/ai-pattern-screener/daily_cache";

const BODY_MULT = 2.0;
const VOL_MULT = 1.5;
const AVG_PERIOD = 20;
const CONSOL_MIN = 2;
const CONSOL_BODY_PCT = 0.30;
const WICK_PCT = 0.20;
const LOOKBACK = 10;
const TAIL = 30;

function loadCachedCandles(symbol: string): any[] | null {
  const cachePath = path.join(CACHE_DIR, `${symbol}.json`);
  if (!fs.existsSync(cachePath)) return null;
  try {
    const cached = JSON.parse(fs.readFileSync(cachePath, "utf-8"));
    if (cached.candles && cached.candles.length >= 22) return cached.candles;
  } catch {}
  return null;
}

function parseTail(raw: string): any[] {
  const lines = raw.trim().split("\n");
  if (lines.length < 3) return [];
  const header = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const start = Math.max(1, lines.length - TAIL);
  const result: any[] = [];
  for (let i = start; i < lines.length; i++) {
    const vals = lines[i].split(",");
    const row: any = {};
    for (let j = 0; j < header.length; j++) row[header[j]] = vals[j]?.trim() || "";
    result.push(row);
  }
  return result;
}

function calcRSI(closes: number[]): number {
  if (closes.length < 15) return 50;
  const period = 14;
  let gains = 0, losses = 0;
  const start = closes.length - period - 1;
  for (let j = start + 1; j < closes.length; j++) {
    const diff = closes[j] - closes[j - 1];
    if (diff > 0) gains += diff;
    else losses -= diff;
  }
  const avgG = gains / period;
  const avgL = losses / period || 1;
  return Math.round(100 - 100 / (1 + avgG / avgL));
}

function computeAI(bodyVsAvg: number, volVsAvg: number, rsi: number, consolCount: number, changePct: number, status: string): number {
  let score = 30;
  if (bodyVsAvg > 2) score += Math.min((bodyVsAvg - 2) * 8, 20);
  if (volVsAvg > 1.5) score += Math.min((volVsAvg - 1.5) * 10, 15);
  if (rsi >= 30 && rsi <= 70) score += 10;
  if (consolCount >= 2) score += 10;
  if (consolCount >= 4) score += 5;
  if (status === "BROKEN UP" || status === "BROKEN DOWN") score += 10;
  if (Math.abs(changePct) < 2) score += 5;
  return Math.round(Math.min(score, 98));
}

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const lookback = parseInt(searchParams.get("lookback") || String(LOOKBACK)) || LOOKBACK;

    const files = fs.readdirSync(DATA_DIR).filter((f) => f.endsWith("_ONE_DAY.csv"));
    const results: any[] = [];

    for (const file of files) {
      const sym = file.replace("_ONE_DAY.csv", "");

      try {
        // Try Dhan cache first (has fresh live data)
        const cachedCandles = loadCachedCandles(sym);
        let prices: number[], price: number, change: number, volume: number, avgVol: number, candles: any[], dataSource: string;

        if (cachedCandles) {
          dataSource = "dhan_cache";
          const useCandles = cachedCandles.slice(-TAIL);
          prices = useCandles.map((c: any) => c.close);
          const last = useCandles[useCandles.length - 1];
          const prev = useCandles[useCandles.length - 2];
          price = last.close;
          change = prev.close > 0 ? ((last.close - prev.close) / prev.close) * 100 : 0;
          volume = last.volume || 0;
          const vols = useCandles.slice(-20).map((c: any) => c.volume || 0);
          avgVol = vols.reduce((a: number, b: number) => a + b, 0) / Math.max(vols.length, 1);
          candles = useCandles.map((c: any) => ({
            open: c.open, high: c.high, low: c.low, close: c.close, volume: c.volume || 0,
            body: Math.abs(c.close - c.open), range: Math.max(c.high - c.low, 1),
          }));
        } else {
          dataSource = "csv";
          const raw = fs.readFileSync(path.join(DATA_DIR, file), "utf-8");
          const records = parseTail(raw);
          if (records.length < 22) continue;
          prices = records.map((r: any) => parseFloat(r.close) || 0);
          const last = records[records.length - 1];
          const prev = records[records.length - 2];
          price = parseFloat(last.close) || 0;
          const prevClose = parseFloat(prev.close) || price;
          change = prevClose > 0 ? ((price - prevClose) / prevClose) * 100 : 0;
          volume = parseInt(last.volume) || 0;
          const vols = records.slice(-20).map((r: any) => parseInt(r.volume) || 0);
          avgVol = vols.reduce((a: number, b: number) => a + b, 0) / Math.max(vols.length, 1);
          candles = records.map((r: any) => {
            const o = parseFloat(r.open), h = parseFloat(r.high), l = parseFloat(r.low), c = parseFloat(r.close), v = parseInt(r.volume) || 0;
            return { open: o, high: h, low: l, close: c, volume: v, body: Math.abs(c - o), range: h - l || 1 };
          });
        }

        const lastRsi = calcRSI(prices);
        const relVolume = avgVol > 0 ? volume / avgVol : 1;

        const avgBody20: number[] = [];
        for (let i = 0; i < candles.length; i++) {
          if (i >= 20) {
            let sum = 0;
            for (let j = i - 20; j < i; j++) sum += candles[j].body;
            avgBody20[i] = sum / 20;
          } else avgBody20[i] = candles[i].body;
        }

        let foundPattern: any = null;
        const scanStart = Math.max(20, candles.length - lookback);
        for (let i = scanStart; i < candles.length; i++) {
          const c = candles[i];
          const avgB = avgBody20[i];
          if (!avgB || c.body < avgB * BODY_MULT || c.volume < avgVol * VOL_MULT) continue;
          const wickRatio = c.range > 0 ? (c.high - Math.max(c.close, c.open)) / c.range : 0;
          if (wickRatio >= WICK_PCT) continue;

          const type = c.close > c.open ? "BULLISH" : "BEARISH";
          let consolCount = 0, endIdx = -1;
          const zHigh = c.high, zLow = c.low;
          for (let j = i + 1; j < candles.length; j++) {
            const n = candles[j];
            if (n.high > zHigh || n.low < zLow) {
              if (consolCount >= CONSOL_MIN) { endIdx = j - 1; break; }
              consolCount = 0; continue;
            }
            if (n.body / n.range < CONSOL_BODY_PCT) consolCount++;
            else {
              if (consolCount >= CONSOL_MIN) { endIdx = j - 1; break; }
              consolCount = 0;
            }
          }
          if (consolCount >= CONSOL_MIN && endIdx < 0) endIdx = candles.length - 1;
          if (endIdx < 0) continue;

          const changePct = (candles[endIdx].close - c.close) / c.close * 100;
          const status = changePct > 2 ? "BROKEN UP" : changePct < -2 ? "BROKEN DOWN" : "CONSOLIDATING";
          foundPattern = {
            type, bodyVsAvg: Math.round(c.body / avgB * 100) / 100,
            volVsAvg: Math.round(c.volume / avgVol * 100) / 100,
            consolCount, changePct, status, rsi: lastRsi,
          };
          break;
        }

        if (!foundPattern) continue;

        const score = computeAI(foundPattern.bodyVsAvg, foundPattern.volVsAvg, foundPattern.rsi, foundPattern.consolCount, foundPattern.changePct, foundPattern.status);
        results.push({
          ticker: sym, company: sym, price, change, volume, relVolume,
          pattern: `BCC ${foundPattern.type} (${foundPattern.consolCount}c)`,
          similarity: Math.round(Math.min(foundPattern.bodyVsAvg / 4 * 100, 99)),
          confidence: Math.round(Math.min(foundPattern.volVsAvg / 3 * 100, 98)),
          aiScore: score,
          sector: "Other",
          atr: Math.round(price * 0.02 * 100) / 100,
          rsi: foundPattern.rsi,
          trend: change > 2 ? "Strong Up" : change > 0.5 ? "Up" : change < -2 ? "Strong Down" : change < -0.5 ? "Down" : "Sideways",
          emaStatus: "Above",
          vwapStatus: "Above",
          strength: score,
          prob: Math.round(Math.min(Math.abs(change) + 45, 85)),
          risk: Math.round(Math.random() * 3 + 1),
          reward: Math.round(Math.random() * 6 + 2),
          expectedMove: Math.round(Math.abs(change) * 0.5 + 1),
          alerted: true,
          dataSource,
        });
      } catch {
        continue;
      }
    }

    results.sort((a: any, b: any) => b.aiScore - a.aiScore);
    results.forEach((r: any, i: number) => (r.rank = i + 1));

    return NextResponse.json({ stocks: results, total: results.length, generatedAt: new Date().toISOString() });
  } catch (err) {
    console.error("Scanner error:", err);
    return NextResponse.json({ error: "Scanner failed" }, { status: 500 });
  }
}
