import { create } from "zustand";

export interface PatternHighlight {
  triggerTime: number;
  triggerType: "BULLISH" | "BEARISH";
  triggerClose: number;
  triggerHigh: number;
  triggerLow: number;
  consolHigh: number;
  consolLow: number;
  consolCount: number;
  consolEndTime: number;
  bodyVsAvg: number;
  volVsAvg: number;
  rsi: number;
  status: string;
  changePct: number;
  consolRangePct: number;
  triggerDate: string;
}

export interface ScannerRow {
  rank: number;
  ticker: string;
  company: string;
  price: number;
  change: number;
  volume: number;
  relVolume: number;
  pattern: string;
  similarity: number;
  confidence: number;
  aiScore: number;
  trend: string;
  sector: string;
  industry: string;
  atr: number;
  rsi: number;
  emaStatus: string;
  vwapStatus: string;
  strength: number;
  prob: number;
  risk: number;
  reward: number;
  expectedMove: number;
  alerted: boolean;
}

export interface Alert {
  id: number;
  type: string;
  message: string;
  time: string;
  severity: "info" | "warning" | "critical";
  read: boolean;
}

export interface BreadthData {
  advancers: number;
  decliners: number;
  newHighs: number;
  newLows: number;
  vix: number;
  topGainers: { ticker: string; price: number; change: number }[];
  topLosers: { ticker: string; price: number; change: number }[];
  mostActive: { ticker: string; volume: number }[];
}

export interface CacheStatus {
  total: number;
  cached: number;
  fresh: number;
  stale: number;
  uncached: number;
  total_candles: number;
  cache_ttl_hours: number;
  oldest_cache: string | null;
  newest_cache: string | null;
}

export interface AIInsight {
  pattern: string;
  reason: string;
  accuracy: number;
  winRate: number;
  risk: number;
  reward: number;
  trend: string;
}

interface AppState {
  sidebarCollapsed: boolean;
  theme: "dark";
  selectedStock: string;
  selectedTimeframe: string;
  patternLength: number;
  similarityThreshold: number;
  scanning: boolean;
  focusPattern: PatternHighlight | null;
  showChartModal: boolean;

  scannerData: ScannerRow[];
  alerts: Alert[];
  unreadAlerts: number;
  globalSearch: string;
  cacheStatus: CacheStatus | null;
  cacheRefreshing: boolean;
  forceRefreshing: boolean;
  breadthData: BreadthData | null;
  aiInsight: AIInsight | null;
  activeSection: string;
  scannerLoading: boolean;
  liveScanning: boolean;
  historyLoading: boolean;
  scannerMode: "live" | "history";
  breadthLoading: boolean;
  compareStock: string;

  toast: { message: string; visible: boolean };
  showToast: (msg: string) => void;

  panelOrder: string[];
  panelVisibility: Record<string, boolean>;

  toggleSidebar: () => void;
  setSelectedStock: (s: string) => void;
  setSelectedTimeframe: (t: string) => void;
  setPatternLength: (n: number) => void;
  setSimilarityThreshold: (n: number) => void;
  setScanning: (b: boolean) => void;
  setScannerMode: (m: "live" | "history") => void;
  setGlobalSearch: (s: string) => void;
  setActiveSection: (s: string) => void;
  setCompareStock: (s: string) => void;
  addAlert: (type: string, message: string, severity: "info" | "warning" | "critical") => void;
  markAlertRead: (id: number) => void;
  clearAlerts: () => void;
  togglePanel: (id: string) => void;
  showPanel: (id: string) => void;
  setPanelOrder: (order: string[]) => void;
  setFocusPattern: (p: PatternHighlight | null) => void;
  setShowChartModal: (b: boolean) => void;
  fetchScanner: () => Promise<void>;
  fetchLiveScanner: () => Promise<void>;
  fetchHistoryScanner: () => Promise<void>;
  fetchBreadth: () => Promise<void>;
  fetchAIInsight: (stock: string) => Promise<void>;
  fetchCacheStatus: () => Promise<void>;
  refreshCache: () => Promise<void>;
  forceRefreshCache: () => Promise<void>;
}

export const useStore = create<AppState>((set, get) => ({
  sidebarCollapsed: false,
  theme: "dark",
  selectedStock: "RELIANCE",
  selectedTimeframe: "1D",
  patternLength: 10,
  similarityThreshold: 85,
  scanning: false,
  focusPattern: null,
  showChartModal: false,

  scannerData: [],
  cacheStatus: null,
  cacheRefreshing: false,
  forceRefreshing: false,
  alerts: [],
  unreadAlerts: 0,
  globalSearch: "",
  breadthData: null,
  aiInsight: null,
  activeSection: "scanner",
  scannerLoading: false,
  liveScanning: false,
  historyLoading: false,
  scannerMode: "live",
  breadthLoading: false,
  compareStock: "ADANIGREEN",

  toast: { message: "", visible: false },
  showToast: (msg) => {
    set({ toast: { message: msg, visible: true } });
    setTimeout(() => set({ toast: { message: "", visible: false } }), 2200);
  },

  panelOrder: ["chart", "insights", "scanner", "dema-atr", "consolidation-breakout", "breadth", "comparison", "builder"],
  panelVisibility: { chart: true, insights: true, scanner: true, "dema-atr": true, "consolidation-breakout": true, breadth: true, comparison: true, builder: false },

  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
  setSelectedStock: (s) => set({ selectedStock: s }),
  setSelectedTimeframe: (t) => set({ selectedTimeframe: t }),
  setPatternLength: (n) => set({ patternLength: n }),
  setSimilarityThreshold: (n) => set({ similarityThreshold: n }),
  setScanning: (b) => set({ scanning: b }),
  setScannerMode: (m) => set({ scannerMode: m }),
  setGlobalSearch: (s) => set({ globalSearch: s }),
  setActiveSection: (s) => set({ activeSection: s }),
  setCompareStock: (s) => set({ compareStock: s }),
  addAlert: (type, message, severity) =>
    set((state) => ({
      alerts: [{ id: Date.now(), type, message, time: "just now", severity, read: false }, ...state.alerts.slice(0, 49)],
      unreadAlerts: state.unreadAlerts + 1,
    })),
  markAlertRead: (id) =>
    set((state) => {
      const updated = state.alerts.map((a) => (a.id === id ? { ...a, read: true } : a));
      return { alerts: updated, unreadAlerts: updated.filter((a) => !a.read).length };
    }),
  clearAlerts: () => set({ alerts: [], unreadAlerts: 0 }),
  togglePanel: (id) =>
    set((state) => ({ panelVisibility: { ...state.panelVisibility, [id]: !state.panelVisibility[id] } })),
  showPanel: (id) =>
    set((state) => ({ panelVisibility: { ...state.panelVisibility, [id]: true } })),
  setPanelOrder: (order) => set({ panelOrder: order }),
  setFocusPattern: (p) => set({ focusPattern: p }),
  setShowChartModal: (b) => set({ showChartModal: b }),

  fetchScanner: async () => {
    set({ scannerLoading: true });
    try {
      const res = await fetch("/api/scanner");
      if (!res.ok) throw new Error("Scanner API failed");
      const data = await res.json();
      set({ scannerData: data.stocks || [] });
    } catch (e) {
      set({ scannerData: [] });
      throw e;
    }
    finally { set({ scannerLoading: false }); }
  },

  fetchLiveScanner: async () => {
    set({ liveScanning: true, scanning: true });
    try {
      const res = await fetch("/api/scanner/live");
      if (!res.ok) throw new Error("Live scanner failed");
      const data = await res.json();
      if (data.stocks) {
        const withRank = data.stocks.map((s: any, i: number) => ({ ...s, rank: i + 1, company: s.ticker }));
        set({ scannerData: withRank });
      }
    } catch { /* keep existing data */ }
    finally { set({ liveScanning: false, scanning: false }); }
  },

  fetchHistoryScanner: async () => {
    set({ historyLoading: true });
    try {
      const res = await fetch("/api/scanner/history");
      if (!res.ok) throw new Error("History scanner failed");
      const data = await res.json();
      if (data.stocks) {
        set({ scannerData: data.stocks });
      } else {
        throw new Error("No history data returned");
      }
    } catch (e) {
      set({ scannerData: [] });
      throw e;
    }
    finally { set({ historyLoading: false }); }
  },

  fetchBreadth: async () => {
    set({ breadthLoading: true });
    try {
      const res = await fetch("/api/market/breadth");
      if (!res.ok) throw new Error("Breadth API failed");
      const data = await res.json();
      set({ breadthData: data });
    } catch { /* keep existing */ }
    finally { set({ breadthLoading: false }); }
  },

  fetchCacheStatus: async () => {
    try {
      const res = await fetch("/api/cache/status");
      if (!res.ok) return;
      const data = await res.json();
      set({ cacheStatus: data });
    } catch {}
  },

  refreshCache: async () => {
    set({ cacheRefreshing: true });
    try {
      await fetch("/api/cache/refresh", { method: "POST" });
      let attempts = 0;
      const poll = setInterval(async () => {
        try {
          const res = await fetch("/api/cache/refresh");
          const data = await res.json();
          if (data.result && (data.result.status === "completed" || data.result.status === "failed")) {
            clearInterval(poll);
            set({ cacheRefreshing: false, cacheStatus: data.result });
          }
        } catch {}
        attempts++;
        if (attempts > 60) { clearInterval(poll); set({ cacheRefreshing: false }); }
      }, 5000);
    } catch {
      set({ cacheRefreshing: false });
    }
  },

  forceRefreshCache: async () => {
    set({ forceRefreshing: true, scanning: true });
    try {
      await fetch("/api/cache/refresh?force=true", { method: "POST" });
      let attempts = 0;
      const poll = setInterval(async () => {
        try {
          const res = await fetch("/api/cache/refresh");
          const data = await res.json();
          if (data.result && (data.result.status === "completed" || data.result.status === "failed")) {
            clearInterval(poll);
            set({ forceRefreshing: false, cacheStatus: data.result });
            // Auto-run scanner after refresh
            get().fetchScanner();
          }
        } catch {}
        attempts++;
        if (attempts > 120) { clearInterval(poll); set({ forceRefreshing: false }); }
      }, 5000);
    } catch {
      set({ forceRefreshing: false });
    }
  },

  fetchAIInsight: async (stock: string) => {
    try {
      const res = await fetch(`/api/stock/${stock}`);
      if (!res.ok) throw new Error("Stock API failed");
      const data = await res.json();
      const patterns = data.patterns || [];
      if (patterns.length > 0) {
        const latest = patterns[patterns.length - 1];
        set({
          aiInsight: {
            pattern: `BCC ${latest.triggerType}`,
            reason: `${latest.triggerDate}: ${latest.triggerType} BCC with ${latest.consolCount}d consolidation. Body ${latest.bodyVsAvg.toFixed(1)}x avg, Vol ${latest.volVsAvg.toFixed(1)}x avg, RSI ${latest.rsi}. Status: ${latest.status}. ${latest.changePct > 0 ? `Post-pattern: +${latest.changePct.toFixed(1)}%` : `Post-pattern: ${latest.changePct.toFixed(1)}%`}`,
            accuracy: computeAccuracy(latest),
            winRate: latest.triggerType === "BULLISH" ? 52 : 48,
            risk: Math.round(Math.random() * 3 + 1),
            reward: Math.round(Math.random() * 5 + 2),
            trend: latest.triggerType === "BULLISH" ? "Bullish" : "Bearish",
          },
        });
      } else {
        set({
          aiInsight: {
            pattern: "No Pattern",
            reason: `No BCC or Squeeze pattern detected for ${stock} in recent data.`,
            accuracy: 0, winRate: 0, risk: 0, reward: 0, trend: "Neutral",
          },
        });
      }
    } catch {
      set({
        aiInsight: {
          pattern: "Error",
          reason: `Could not fetch data for ${stock}.`,
          accuracy: 0, winRate: 0, risk: 0, reward: 0, trend: "Neutral",
        },
      });
    }
  },
}));

function computeAccuracy(p: PatternHighlight | any): number {
  let score = 50;
  if (p.bodyVsAvg > 2) score += 10;
  if (p.bodyVsAvg > 3) score += 5;
  if (p.volVsAvg > 1.5) score += 10;
  if (p.volVsAvg > 2.5) score += 5;
  if (p.consolCount >= 3) score += 5;
  if (p.consolCount >= 5) score += 5;
  if (p.rsi >= 30 && p.rsi <= 70) score += 5;
  if (p.status === "BROKEN UP") score += 10;
  if (p.status === "BROKEN DOWN") score += 10;
  if (Math.abs(p.changePct) < 2) score += 5;
  return Math.min(score, 98);
}
