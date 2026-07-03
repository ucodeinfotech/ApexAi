import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { parse } from "csv-parse/sync";
import { spawn } from "child_process";
import { exec } from "child_process";
import { promisify } from "util";
const execAsync = promisify(exec);

const DATA_DIR = "C:/Users/pc/Downloads/stock hist data/comprehensive_data";
const CACHE_DIR = "C:/Users/pc/Downloads/stock hist data/ai-pattern-screener/daily_cache";
const CACHE_SCRIPT = "C:/Users/pc/Downloads/stock hist data/ai-pattern-screener/dhan_cache_manager.py";
const CACHE_TTL = 24 * 3600 * 1000;

interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  body: number;
  range: number;
  avgBody20?: number;
  avgVol20?: number;
  rsi14?: number;
}

function calcRSI(closes: number[], period = 14): number[] {
  const rsis: number[] = new Array(closes.length).fill(50);
  for (let i = period; i < closes.length; i++) {
    let gains = 0, losses = 0;
    for (let j = i - period + 1; j <= i; j++) {
      const diff = closes[j] - closes[j - 1];
      if (diff > 0) gains += diff;
      else losses -= diff;
    }
    const avgG = gains / period;
    const avgL = losses / period || 1;
    const rs = avgG / avgL;
    rsis[i] = 100 - 100 / (1 + rs);
  }
  return rsis;
}

export interface BCCPattern {
  triggerDate: string;
  triggerTime: number;
  triggerType: "BULLISH" | "BEARISH";
  triggerClose: number;
  triggerBody: number;
  triggerHigh: number;
  triggerLow: number;
  bodyVsAvg: number;
  volVsAvg: number;
  wickPct: number;
  rsi: number;
  consolCount: number;
  consolHigh: number;
  consolLow: number;
  consolRangePct: number;
  consolEndTime: number;
  status: string;
  changePct: number;
}

interface StockResponse {
  candles: Candle[];
  patterns: BCCPattern[];
  symbol: string;
  dataSource: string;
}

// BCC Scanner params
const BODY_MULT = 2.0;
const VOL_MULT = 1.5;
const WICK_PCT = 0.20;
const AVG_PERIOD = 20;
const CONSOL_MIN = 2;
const CONSOL_BODY_PCT = 0.30;

function detectBCC(candles: Candle[]): BCCPattern[] {
  const patterns: BCCPattern[] = [];
  for (let i = AVG_PERIOD; i < candles.length; i++) {
    const c = candles[i];
    const avgBody = c.avgBody20 || 0;
    const avgVol = c.avgVol20 || 0;
    if (!avgBody || !avgVol) continue;
    const wickRatio = (c.high - Math.max(c.close, c.open)) / c.range;
    if (c.body < avgBody * BODY_MULT || c.volume < avgVol * VOL_MULT || wickRatio >= WICK_PCT) continue;

    const triggerType = c.close > c.open ? "BULLISH" as const : "BEARISH" as const;
    let consolCount = 0, consolEndIdx = -1;
    const zoneHigh = c.high, zoneLow = c.low;

    for (let j = i + 1; j < candles.length; j++) {
      const n = candles[j];
      if (n.high > zoneHigh || n.low < zoneLow) {
        if (consolCount >= CONSOL_MIN) { consolEndIdx = j - 1; break; }
        consolCount = 0; continue;
      }
      if (n.body / n.range < CONSOL_BODY_PCT) consolCount++;
      else {
        if (consolCount >= CONSOL_MIN) { consolEndIdx = j - 1; break; }
        consolCount = 0;
      }
    }
    if (consolCount >= CONSOL_MIN && consolEndIdx < 0) consolEndIdx = candles.length - 1;
    if (consolEndIdx < 0) continue;

    const changePct = (candles[consolEndIdx].close - c.close) / c.close * 100;
    const status = changePct > 2 ? "BROKEN UP" : changePct < -2 ? "BROKEN DOWN" : "CONSOLIDATING";
    patterns.push({
      triggerDate: new Date(c.time * 1000).toISOString().split("T")[0],
      triggerTime: c.time, triggerType, triggerClose: c.close,
      triggerBody: c.body, triggerHigh: c.high, triggerLow: c.low,
      bodyVsAvg: Math.round(c.body / avgBody * 100) / 100,
      volVsAvg: Math.round(c.volume / avgVol * 100) / 100,
      wickPct: Math.round(wickRatio * 1000) / 10, rsi: Math.round(c.rsi14 || 50),
      consolCount, consolHigh: Math.round(zoneHigh * 100) / 100,
      consolLow: Math.round(zoneLow * 100) / 100,
      consolRangePct: Math.round((zoneHigh - zoneLow) / c.close * 10000) / 100,
      consolEndTime: candles[consolEndIdx].time, status, changePct: Math.round(changePct * 100) / 100,
    });
  }
  return patterns;
}

function computeCandleIndicators(candles: Candle[]): void {
  const closes = candles.map((c) => c.close);
  const rsis = calcRSI(closes, 14);
  for (let i = 0; i < candles.length; i++) {
    if (i >= AVG_PERIOD) {
      let sumBody = 0, sumVol = 0;
      for (let j = i - AVG_PERIOD; j < i; j++) { sumBody += candles[j].body; sumVol += candles[j].volume; }
      candles[i].avgBody20 = sumBody / AVG_PERIOD;
      candles[i].avgVol20 = sumVol / AVG_PERIOD;
    }
    candles[i].rsi14 = rsis[i];
  }
}

function ema(values: number[], period: number): number[] {
  const result: number[] = new Array(values.length).fill(values[0]);
  const k = 2 / (period + 1);
  for (let i = 0; i < values.length; i++) {
    if (i > 0) result[i] = values[i] * k + result[i - 1] * (1 - k);
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

function computeDemaAtr(candles: Candle[], demaPeriod = 7, atrPeriod = 14, atrFactor = 1.7): number[] {
  const closes = candles.map((c) => c.close);
  const ema1 = ema(closes, demaPeriod);
  const ema2 = ema(ema1, demaPeriod);
  const dema: number[] = [];
  for (let i = 0; i < closes.length; i++) dema[i] = 2 * ema1[i] - ema2[i];

  const tr: number[] = [];
  for (let i = 0; i < candles.length; i++) {
    if (i === 0) { tr.push(candles[i].range); continue; }
    tr.push(Math.max(candles[i].range, Math.abs(candles[i].high - candles[i - 1].close), Math.abs(candles[i].low - candles[i - 1].close)));
  }
  const atrVals = rma(tr, atrPeriod);

  const result: number[] = new Array(closes.length).fill(0);
  for (let i = 0; i < closes.length; i++) {
    const band = atrVals[i] * atrFactor;
    const upper = dema[i] + band;
    const lower = dema[i] - band;
    if (i === 0) result[i] = dema[i];
    else {
      result[i] = result[i - 1];
      if (lower > result[i]) result[i] = lower;
      if (upper < result[i]) result[i] = upper;
    }
  }
  return result;
}

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await params;
  const sym = symbol.toUpperCase();
  const filePath = path.join(DATA_DIR, `${sym}_ONE_DAY.csv`);
  let candles: Candle[] = [];

  // ─── Try Dhan cache first (live data, ~100 candles) ───
  const cachePath = path.join(CACHE_DIR, `${sym}.json`);
  let cacheValid = false;
  if (fs.existsSync(cachePath)) {
    const age = Date.now() - fs.statSync(cachePath).mtimeMs;
    if (age < CACHE_TTL) cacheValid = true;
  }
  if (cacheValid) {
    try {
      const cached = JSON.parse(fs.readFileSync(cachePath, "utf-8"));
      if (cached.candles && cached.candles.length >= 22) {
        candles = cached.candles.map((r: any) => ({
          time: r.time, open: r.open, high: r.high, low: r.low,
          close: r.close, volume: r.volume || 0,
          body: Math.abs(r.close - r.open), range: Math.max(r.high - r.low, 1),
        }));
      }
    } catch {}
  }

  // ─── Fetch today's candle if cache doesn't have it ───
  if (candles.length > 0) {
    const todayStart = Math.floor(new Date().setHours(0,0,0,0) / 1000);
    const lastCandleTime = candles[candles.length - 1].time;
    if (lastCandleTime < todayStart) {
      let todayCandle: any = null;
      // Try 1: historical API
      try {
        const { stdout } = await execAsync(`python "${CACHE_SCRIPT}" today ${sym}`, { timeout: 15000 });
        const todayData = JSON.parse(stdout);
        if (todayData.candles && todayData.candles.length > 0) {
          const c = todayData.candles[todayData.candles.length - 1];
          if (c.time > lastCandleTime) todayCandle = c;
        }
      } catch {}
      // Try 2: WebSocket live data (fallback for intraday)
      if (!todayCandle) {
        try {
          const { stdout } = await execAsync(`python "${CACHE_SCRIPT}" live ${sym}`, { timeout: 15000 });
          const liveData = JSON.parse(stdout);
          if (liveData.quote) {
            const q = liveData.quote;
            todayCandle = {
              time: todayStart,
              date: new Date().toISOString().split("T")[0],
              open: q.open, high: q.high, low: q.low,
              close: q.ltp, volume: q.volume,
            };
          }
        } catch {}
      }
      if (todayCandle) {
        candles.push({
          time: todayCandle.time, open: todayCandle.open, high: todayCandle.high,
          low: todayCandle.low, close: todayCandle.close, volume: todayCandle.volume || 0,
          body: Math.abs(todayCandle.close - todayCandle.open),
          range: Math.max(todayCandle.high - todayCandle.low, 1),
        });
        // Update cache
        try {
          const cached = JSON.parse(fs.readFileSync(cachePath, "utf-8"));
          // Avoid duplicate if same time already exists
          const dup = cached.candles.find((c: any) => c.time === todayCandle.time);
          if (!dup) {
            cached.candles.push(todayCandle);
            cached.count = cached.candles.length;
          }
          cached.cached_at = new Date().toISOString();
          fs.writeFileSync(cachePath, JSON.stringify(cached, null, 2));
        } catch {}
      }
    }
  }

  // Spawn background caching if no valid cache
  if (!cacheValid && fs.existsSync(path.join(DATA_DIR, `${sym}_ONE_DAY.csv`))) {
    const cp = spawn("python", [CACHE_SCRIPT, "fetch", sym], {
      stdio: "ignore", detached: true,
    });
    cp.unref();
  }

  // ─── Fallback to CSV data (historical, ~3000+ candles) ───
  if (candles.length === 0) {
    if (!fs.existsSync(filePath)) {
      return NextResponse.json({ error: "Stock not found" }, { status: 404 });
    }
    try {
      const raw = fs.readFileSync(filePath, "utf-8");
      const records = parse(raw, { columns: true, skip_empty_lines: true, trim: true });
      candles = records.map((r: any) => {
        const dt = new Date(r.datetime);
        const o = parseFloat(r.open), h = parseFloat(r.high), l = parseFloat(r.low), c = parseFloat(r.close), v = parseInt(r.volume) || 0;
        return { time: Math.floor(dt.getTime() / 1000), open: o, high: h, low: l, close: c, volume: v, body: Math.abs(c - o), range: h - l || 1 };
      });
    } catch (err) {
      console.error(err);
      return NextResponse.json({ error: "Failed to parse stock data" }, { status: 500 });
    }
  }

  computeCandleIndicators(candles);
  const patterns = detectBCC(candles);
  const demaAtr = computeDemaAtr(candles);

  return NextResponse.json({
    candles: candles.slice(-300),
    demaAtr: demaAtr.slice(-300),
    patterns,
    symbol: sym,
    dataSource: fs.existsSync(cachePath) ? "dhan_cache" : "csv",
  } satisfies StockResponse & { dataSource: string });
}
