import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const DATA_DIR = "C:/Users/pc/Downloads/stock hist data/comprehensive_data";
const CACHE_DIR = "C:/Users/pc/Downloads/stock hist data/ai-pattern-screener/daily_cache";

const DEMA_PERIOD = 7;
const ATR_PERIOD = 14;
const ATR_FACTOR = 1.7;
const PRICE_NEAR_PCT = 2;

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

function ema(values: number[], period: number): number[] {
  const result: number[] = new Array(values.length).fill(values[0]);
  const k = 2 / (period + 1);
  for (let i = 0; i < values.length; i++) {
    if (i === 0) result[i] = values[i];
    else result[i] = values[i] * k + result[i - 1] * (1 - k);
  }
  return result;
}

function rma(values: number[], period: number): number[] {
  const result: number[] = new Array(values.length).fill(values[0]);
  for (let i = 0; i < values.length; i++) {
    if (i === 0) result[i] = values[i];
    else result[i] = (values[i] + (period - 1) * result[i - 1]) / period;
  }
  return result;
}

function atr(candles: { high: number; low: number; close: number }[], period: number): number[] {
  const tr: number[] = [];
  for (let i = 0; i < candles.length; i++) {
    if (i === 0) { tr.push(candles[i].high - candles[i].low); continue; }
    const h = candles[i].high;
    const l = candles[i].low;
    const pc = candles[i - 1].close;
    tr.push(Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc)));
  }
  return rma(tr, period);
}

function computeDemaAtr(closes: number[], atrValues: number[], period: number, factor: number): number[] {
  const ema1 = ema(closes, period);
  const ema2 = ema(ema1, period);
  const dema: number[] = [];
  for (let i = 0; i < closes.length; i++) {
    dema[i] = 2 * ema1[i] - ema2[i];
  }
  const demaAtr: number[] = new Array(closes.length).fill(0);
  for (let i = 0; i < closes.length; i++) {
    const tr = atrValues[i] * factor;
    const upper = dema[i] + tr;
    const lower = dema[i] - tr;
    if (i === 0) {
      demaAtr[i] = dema[i];
    } else {
      demaAtr[i] = demaAtr[i - 1];
      if (lower > demaAtr[i]) demaAtr[i] = lower;
      if (upper < demaAtr[i]) demaAtr[i] = upper;
    }
  }
  return demaAtr;
}

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const lookback = parseInt(searchParams.get("lookback") || "30") || 30;

    const files = fs.readdirSync(DATA_DIR).filter((f) => f.endsWith("_ONE_DAY.csv"));
    const results: any[] = [];

    for (const file of files) {
      const sym = file.replace("_ONE_DAY.csv", "");

      try {
        const cachedCandles = loadCachedCandles(sym);
        let closes: number[], highs: number[], lows: number[], volumes: number[], price: number;
        let candles: { open: number; high: number; low: number; close: number }[];

        if (cachedCandles) {
          const use = cachedCandles.slice(-lookback);
          closes = use.map((c: any) => c.close);
          highs = use.map((c: any) => c.high);
          lows = use.map((c: any) => c.low);
          volumes = use.map((c: any) => c.volume || 0);
          price = closes[closes.length - 1];
          candles = use;
        } else {
          const raw = fs.readFileSync(path.join(DATA_DIR, file), "utf-8");
          const records = parseTail(raw, lookback);
          if (records.length < 22) continue;
          closes = records.map((r: any) => parseFloat(r.close) || 0);
          highs = records.map((r: any) => parseFloat(r.high) || 0);
          lows = records.map((r: any) => parseFloat(r.low) || 0);
          volumes = records.map((r: any) => parseInt(r.volume) || 0);
          price = closes[closes.length - 1];
          candles = records.map((r: any) => ({
            open: parseFloat(r.open) || 0, high: parseFloat(r.high) || 0,
            low: parseFloat(r.low) || 0, close: parseFloat(r.close) || 0,
          }));
        }

        if (closes.length < DEMA_PERIOD + ATR_PERIOD + 2) continue;

        const atrValues = atr(candles, ATR_PERIOD);
        const demaAtr = computeDemaAtr(closes, atrValues, DEMA_PERIOD, ATR_FACTOR);

        const cur = demaAtr[demaAtr.length - 1];
        const prev = demaAtr.length >= 2 ? demaAtr[demaAtr.length - 2] : cur;
        const slope = cur - prev;

        const prev5 = demaAtr.length >= 6 ? demaAtr[demaAtr.length - 6] : prev;
        const slope5 = cur - prev5;

        const prev15 = demaAtr.length >= 16 ? demaAtr[demaAtr.length - 16] : demaAtr[0];
        const slope15 = cur - prev15;

        const lineRising = slope15 > 0;
        const direction = lineRising ? "UP" : "DOWN";
        const priceDistPct = price > 0 ? Math.abs((price - cur) / cur) * 100 : 999;
        const nearLine = priceDistPct <= PRICE_NEAR_PCT;

        if (!nearLine) continue;

        results.push({
          ticker: sym,
          price: Math.round(price * 100) / 100,
          demaAtr: Math.round(cur * 100) / 100,
          direction,
          slope: Math.round(slope * 10000) / 10000,
          slope15: Math.round(slope15 * 10000) / 10000,
          slope5: Math.round(slope5 * 10000) / 10000,
          priceDistPct: Math.round(priceDistPct * 100) / 100,
          volume: volumes.reduce((a, b) => a + b, 0),
        });
      } catch {
        continue;
      }
    }

    // Sort: strongest absolute slope first (mix of UP and DOWN)
    results.sort((a, b) => Math.abs(b.slope5) - Math.abs(a.slope5));
    results.forEach((r, i) => (r.rank = i + 1));

    return NextResponse.json({ stocks: results, total: results.length, generatedAt: new Date().toISOString() });
  } catch (err) {
    console.error("DEMA ATR scanner error:", err);
    return NextResponse.json({ error: "DEMA ATR scanner failed" }, { status: 500 });
  }
}
