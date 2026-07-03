export function cn(...classes: (string | undefined | null | false)[]): string {
  return classes.filter(Boolean).join(" ");
}

export function formatVolume(v: number): string {
  if (v >= 1_000_000_000) return (v / 1_000_000_000).toFixed(1) + "B";
  if (v >= 1_000_000) return (v / 1_000_000).toFixed(1) + "M";
  if (v >= 1_000) return (v / 1_000).toFixed(1) + "K";
  return v.toString();
}

export function formatPercent(v: number): string {
  return (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
}

export function randomBetween(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

export function truncate(str: string, len: number): string {
  return str.length > len ? str.slice(0, len) + "..." : str;
}

export const indianStocks = [
  "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC", "SBIN",
  "BHARTIARTL", "KOTAKBANK", "BAJFINANCE", "LT", "WIPRO", "AXISBANK", "TITAN",
  "ADANIENT", "ASIANPAINT", "MARUTI", "SUNPHARMA", "HCLTECH", "NTPC", "ONGC",
  "POWERGRID", "ULTRACEMCO", "BAJAJFINSV", "ADANIPORTS", "NESTLEIND", "M&M",
  "JSWSTEEL", "TECHM", "HDFCLIFE", "SBILIFE", "TATASTEEL", "COALINDIA", "BPCL",
  "GRASIM", "INDUSINDBK", "DRREDDY", "CIPLA", "BRITANNIA", "EICHERMOT", "HINDALCO",
  "TATAMOTORS", "APOLLOHOSP", "DIVISLAB", "BAJAJ-AUTO", "SBICARD", "BEL", "IOC",
  "HEROMOTOCO", "GAIL", "HAL", "PIDILITIND", "DLF", "ICICIPRULI", "VEDL",
  "BANDHANBNK", "MARICO", "TORNTPHARM", "SHREECEM", "SIEMENS", "HINDZINC",
  "ATUL", "ABFRL", "NAUKRI", "DABUR", "COLPAL", "BERGEPAINT", "AMBUJACEM",
  "MCDOWELL-N", "HAVELLS", "BANKBARODA", "LUPIN", "BHARATFORG", "SRTRANSFIN",
  "ICICIGI", "MUTHOOTFIN", "CADILAHC", "JUBLFOOD", "INDIGO", "TVSMOTOR",
  "GODREJCP", "PAGEIND", "ASTRAL", "TRENT", "POLYCAB", "DMART", "ICRA",
  "ADANIGREEN", "ADANITRANS", "PNB", "YESBANK", "IDFCFIRSTB", "IEX", "ZOMATO",
  "PAYTM", "LIC", "IRCTC", "INOXLEISUR", "PVRINOX", "BIOCON", "IDEA",
  "ASHOKLEY", "BHEL", "CANBK", "RECLTD", "PFC", "SAIL", "NHPC", "IRFC",
  "HUDCO", "NBCC", "ITI", "GMRINFRA", "MMTC", "COFCOEE", "RBLBANK",
];

export const patternNames = [
  "Bull Flag", "Bear Flag", "Ascending Triangle", "Descending Triangle",
  "Symmetrical Triangle", "Falling Wedge", "Rising Wedge", "Double Bottom",
  "Double Top", "Head & Shoulders", "Inverse H&S", "Cup & Handle",
  "VCP Contraction", "Bollinger Squeeze", "Volume Breakout",
  "Breakaway Gap", "Exhaustion Gap", "Island Reversal", "Morning Star",
  "Evening Star", "Three White Soldiers", "Three Black Crows",
  "Hammer", "Shooting Star", "Doji", "Engulfing", "Harami",
  "Marubozu", "Spinning Top", "Piercing Line", "Dark Cloud Cover",
  "Abandoned Baby", "Kicker", "Three Inside Up", "Three Inside Down",
  "Rising Three Methods", "Falling Three Methods", "Tasuki Gap",
  "Upside Gap Three", "Downside Gap Three", "Big Candle + Consolidation",
  "Squeeze Breakout", "DEMA Squeeze", "VWAP Reclaim", "ATR Contraction",
  "Raff Channel Breakout", "Darvas Box", "Wyckoff Accumulation",
  "Wyckoff Distribution", "Spring", "Upthrust", "SOS", "SOW",
  "Liquidity Grab", "Order Block Break", "FVG Fill", "MSS / CHoCH",
  "Implied Fair Value Gap", "ICT Killzone", "Liquidity Sweep",
  "Institutional Supply", "Institutional Demand",
];

export const sectors = [
  "Banking", "IT", "Pharma", "Auto", "FMCG", "Energy", "Metal", "Infra",
  "Telecom", "Insurance", "Consumer", "Chemicals", "Power", "Realty",
  "Textile", "Media", "Hospitality", "Logistics", "Healthcare", "Education",
];

export const industries = [
  "Private Bank", "Public Bank", "Software", "Consulting", "Pharma", "Biotech",
  "4-Wheeler", "2-Wheeler", "Auto Parts", "Diversified", "Cigarettes",
  "Food", "Beverages", "Refining", "Exploration", "Steel", "Aluminum",
  "Cement", "Construction", "Telecom Services", "Life Insurance",
  "Health Insurance", "Chemicals", "Fertilizers", "Power Generation",
  "Power Trading", "Real Estate", "Textiles", "Media", "Entertainment",
  "Hotel", "Logistics", "Hospital", "Education", "Retail", "E-Commerce",
];

export const timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1D", "1W", "1M"];

export const patternLengths = [5, 8, 10, 13, 15, 20, 25, 30, 50, 100];

export function generateMockScannerData(count = 100) {
  const data = [];
  const shuffled = [...indianStocks].sort(() => Math.random() - 0.5);
  for (let i = 0; i < Math.min(count, shuffled.length); i++) {
    const price = randomBetween(50, 3500);
    const change = randomBetween(-5, 5);
    const vol = randomBetween(100_000, 50_000_000);
    const pattern = patternNames[Math.floor(Math.random() * patternNames.length)];
    const sim = randomBetween(65, 99);
    const conf = randomBetween(55, 98);
    const aiScore = randomBetween(40, 95);
    const atr = price * randomBetween(0.01, 0.04);
    data.push({
      rank: i + 1,
      ticker: shuffled[i],
      company: shuffled[i],
      price: +price.toFixed(2),
      change: +change.toFixed(2),
      volume: Math.round(vol),
      relVolume: +randomBetween(0.3, 3.5).toFixed(2),
      pattern,
      similarity: +sim.toFixed(1),
      confidence: +conf.toFixed(1),
      aiScore: +aiScore.toFixed(1),
      trend: change > 2 ? "Strong Up" : change > 0.5 ? "Up" : change < -2 ? "Strong Down" : change < -0.5 ? "Down" : "Sideways",
      sector: sectors[Math.floor(Math.random() * sectors.length)],
      industry: industries[Math.floor(Math.random() * industries.length)],
      atr: +atr.toFixed(2),
      rsi: +randomBetween(20, 80).toFixed(0),
      emaStatus: Math.random() > 0.5 ? "Above" : "Below",
      vwapStatus: Math.random() > 0.5 ? "Above" : "Below",
      strength: +randomBetween(20, 90).toFixed(0),
      prob: +randomBetween(45, 85).toFixed(0),
      risk: +randomBetween(1, 5).toFixed(1),
      reward: +randomBetween(2, 12).toFixed(1),
      expectedMove: +randomBetween(1, 8).toFixed(1),
      alerted: Math.random() > 0.7,
    });
  }
  return data.sort((a, b) => b.aiScore - a.aiScore);
}

export function generateMockCandles(count = 200) {
  const data = [];
  let price = 2500 + Math.random() * 500;
  const now = Date.now();
  for (let i = count; i >= 0; i--) {
    const change = (Math.random() - 0.48) * price * 0.025;
    const open = price;
    const close = price + change;
    const high = Math.max(open, close) + Math.random() * Math.abs(change) * 0.5;
    const low = Math.min(open, close) - Math.random() * Math.abs(change) * 0.5;
    data.push({
      time: (now - i * 86400000) / 1000,
      open: +open.toFixed(2),
      high: +high.toFixed(2),
      low: +low.toFixed(2),
      close: +close.toFixed(2),
      volume: Math.round(randomBetween(500_000, 15_000_000)),
    });
    price = close;
  }
  return data;
}

export function generateMarketBreadth() {
  return {
    advancers: Math.round(randomBetween(800, 1800)),
    decliners: Math.round(randomBetween(400, 1200)),
    newHighs: Math.round(randomBetween(20, 200)),
    newLows: Math.round(randomBetween(5, 80)),
    putCall: +randomBetween(0.7, 1.4).toFixed(2),
    vix: +randomBetween(11, 25).toFixed(2),
    sectorLeaders: sectors.slice(0, 5).map((s) => ({
      name: s,
      change: +randomBetween(0.5, 3).toFixed(2),
    })),
    sectorLosers: sectors.slice(5, 10).map((s) => ({
      name: s,
      change: +randomBetween(-3, -0.5).toFixed(2),
    })),
    topGainers: indianStocks.slice(0, 5).map((s) => ({
      ticker: s,
      price: +randomBetween(100, 3000).toFixed(2),
      change: +randomBetween(3, 10).toFixed(2),
    })),
    topLosers: indianStocks.slice(5, 10).map((s) => ({
      ticker: s,
      price: +randomBetween(100, 3000).toFixed(2),
      change: +randomBetween(-10, -3).toFixed(2),
    })),
    mostActive: indianStocks.slice(0, 5).map((s) => ({
      ticker: s,
      volume: Math.round(randomBetween(5_000_000, 50_000_000)),
    })),
  };
}

export function generateHeatmapData() {
  return sectors.map((sector) => ({
    name: sector,
    change: +randomBetween(-4, 5).toFixed(2),
    volume: Math.round(randomBetween(1e8, 5e9)),
    stocks: indianStocks.slice(0, 5).map((s) => ({
      ticker: s,
      change: +randomBetween(-6, 8).toFixed(2),
      volume: Math.round(randomBetween(1e6, 3e7)),
    })),
  }));
}

export function generateAIInsight() {
  const insights = [
    {
      pattern: "Bull Flag",
      reason: "Higher lows, strong volume compression, EMA alignment bullish, VWAP holding as support, ATR compression indicating imminent expansion, momentum shift detected on RSI(14) crossing above 50.",
      accuracy: 72.4,
      winRate: 68.2,
      risk: 2.3,
      reward: 5.8,
      trend: "Bullish",
    },
    {
      pattern: "Big Candle + Consolidation",
      reason: "Volume spike 2.8x average, small-bodied consolidation forming within 3.2% of trigger, RSI cooling from 72 to 58, volume declining 62% during consolidation — coiled spring setup.",
      accuracy: 54.8,
      winRate: 51.3,
      risk: 2.8,
      reward: 4.1,
      trend: "Bullish",
    },
    {
      pattern: "Squeeze Breakout",
      reason: "Bollinger Band width at 6-month low, 7 consecutive low-volatility days, sudden volume expansion 3.2x average, breakout above resistance with conviction, institutional bid detected.",
      accuracy: 61.2,
      winRate: 57.6,
      risk: 2.5,
      reward: 5.2,
      trend: "Bullish",
    },
  ];
  return insights[Math.floor(Math.random() * insights.length)];
}

export function generateAlerts() {
  const types = ["Pattern", "Volume", "Breakout", "Risk", "Market"];
  const severities: ("info" | "warning" | "critical")[] = ["info", "warning", "critical"];
  return Array.from({ length: 8 }, (_, i) => ({
    id: i,
    type: types[Math.floor(Math.random() * types.length)],
    message: `${indianStocks[Math.floor(Math.random() * indianStocks.length)]} — ${patternNames[Math.floor(Math.random() * patternNames.length)]} detected`,
    time: `${Math.floor(Math.random() * 59)}m ago`,
    severity: severities[Math.floor(Math.random() * severities.length)],
    read: Math.random() > 0.4,
  }));
}
