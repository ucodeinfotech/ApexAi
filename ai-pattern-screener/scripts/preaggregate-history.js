const fs = require("fs");
const path = require("path");

const SEEN_FILE = path.resolve(__dirname, "../../backtest_results/seen_patterns_bcc.csv");
const OUTPUT_FILE = path.resolve(__dirname, "../src/app/api/scanner/history/aggregated.json");

const sectorMap = {
  HDFCBANK:"Banking",ICICIBANK:"Banking",SBIN:"Banking",KOTAKBANK:"Banking",AXISBANK:"Banking",INDUSINDBK:"Banking",BANDHANBNK:"Banking",FEDERALBNK:"Banking",RBLBANK:"Banking",IDFCFIRSTB:"Banking",PNB:"Banking",BANKBARODA:"Banking",CANBK:"Banking",INDIANB:"Banking",MAHABANK:"Banking",YESBANK:"Banking",IDBI:"Banking",
  TCS:"IT",INFY:"IT",WIPRO:"IT",HCLTECH:"IT",TECHM:"IT",MPHASIS:"IT",COFORGE:"IT",PERSISTENT:"IT",LTTS:"IT",BSOFT:"IT",KPITTECH:"IT",
  SUNPHARMA:"Pharma",DRREDDY:"Pharma",CIPLA:"Pharma",DIVISLAB:"Pharma",LUPIN:"Pharma",ALKEM:"Pharma",MANKIND:"Pharma",TORNTPHARM:"Pharma",
  MARUTI:"Auto",M_M:"Auto",TATAMOTORS:"Auto",EICHERMOT:"Auto",HEROMOTOCO:"Auto",TVSMOTOR:"Auto",
  RELIANCE:"Energy",ONGC:"Energy",BPCL:"Energy",IOC:"Energy",HINDPETRO:"Energy",GAIL:"Energy",
  NTPC:"Power",POWERGRID:"Power",TATAPOWER:"Power",
  HINDALCO:"Metal",JSWSTEEL:"Metal",TATASTEEL:"Metal",COALINDIA:"Metal",SAIL:"Metal",JINDALSTEL:"Metal",NMDC:"Metal",
  BHARTIARTL:"Telecom",IDEA:"Telecom",
  DLF:"Realty",OBEROIRLTY:"Realty",PRESTIGE:"Realty",GODREJPROP:"Realty",
};

function computeScore(bodyVsAvg, volVsAvg, rsi, consolCount) {
  let s = 30;
  if (bodyVsAvg > 2) s += Math.min((bodyVsAvg - 2) * 8, 20);
  if (volVsAvg > 1.5) s += Math.min((volVsAvg - 1.5) * 10, 15);
  if (rsi >= 30 && rsi <= 70) s += 10;
  if (consolCount >= 3) s += 10;
  if (consolCount >= 5) s += 5;
  return Math.round(Math.min(s, 98));
}

if (!fs.existsSync(SEEN_FILE)) {
  console.error("ERROR: seen_patterns_bcc.csv not found at", SEEN_FILE);
  process.exit(1);
}

console.log("Reading", SEEN_FILE, "...");
const raw = fs.readFileSync(SEEN_FILE, "utf-8");
const lines = raw.trim().split("\n");
const header = lines[0].split(",").map((h) => h.trim());

console.log(`Parsing ${lines.length - 1} records...`);
const records = [];
for (let i = 1; i < lines.length; i++) {
  const vals = lines[i].split(",");
  const row = {};
  for (let j = 0; j < header.length; j++) {
    row[header[j]] = vals[j]?.trim() || "";
  }
  records.push(row);
}

// Group patterns by stock
const stockPatterns = new Map();
for (const r of records) {
  const sym = r.symbol;
  if (!stockPatterns.has(sym)) stockPatterns.set(sym, []);
  stockPatterns.get(sym).push(r);
}

console.log(`Aggregating ${stockPatterns.size} stocks...`);
const results = [];

for (const [sym, patterns] of stockPatterns) {
  const totalPatterns = patterns.length;
  patterns.sort((a, b) => b.trigger_date.localeCompare(a.trigger_date));
  const latest = patterns[0];

  const price = parseFloat(latest.last_close) || 0;
  const bullish = patterns.filter((p) => p.trigger_type === "BULLISH").length;
  const bearish = patterns.filter((p) => p.trigger_type === "BEARISH").length;
  const brokenUp = patterns.filter((p) => p.status === "BROKEN UP").length;
  const brokenDown = patterns.filter((p) => p.status === "BROKEN DOWN").length;
  const consolidating = patterns.filter((p) => p.status === "CONSOLIDATING").length;
  const successRate = totalPatterns > 0 ? Math.round((brokenUp / totalPatterns) * 100) : 0;

  const bv = parseFloat(latest.body_vs_avg) || 0;
  const vv = parseFloat(latest.vol_vs_avg) || 0;
  const score = computeScore(bv, vv, parseInt(latest.rsi) || 50, parseInt(latest.consol_candles) || 0);

  results.push({
    ticker: sym, company: sym,
    price: Math.round(price * 100) / 100,
    change: 0,
    pattern: `BCC ${latest.trigger_type}`,
    patternCount: totalPatterns,
    bullishCount: bullish, bearishCount: bearish,
    brokenUp, brokenDown, consolidating,
    successRate,
    aiScore: score,
    latestDate: latest.trigger_date,
    latestStatus: latest.status,
    bodyVsAvg: bv, volVsAvg: vv,
    rsi: parseInt(latest.rsi) || 50,
    consolCandles: parseInt(latest.consol_candles) || 0,
    sector: sectorMap[sym] || "Other",
    volume: 0, relVolume: 0, similarity: 0, confidence: 0,
    trend: latest.trigger_type === "BULLISH" ? "Bullish" : "Bearish",
    alerted: true,
  });
}

results.sort((a, b) => b.aiScore - a.aiScore);
results.forEach((r, i) => (r.rank = i + 1));

const output = { stocks: results, total: results.length, generatedAt: new Date().toISOString() };
fs.mkdirSync(path.dirname(OUTPUT_FILE), { recursive: true });
fs.writeFileSync(OUTPUT_FILE, JSON.stringify(output, null, 2));

console.log(`✓ Aggregated ${results.length} stocks → ${OUTPUT_FILE}`);
console.log(`  Total patterns: ${records.length}`);
