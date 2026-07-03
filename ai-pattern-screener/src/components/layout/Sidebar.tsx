"use client";

import { useStore } from "@/store/useStore";
import {
  LayoutDashboard, Radar, Puzzle, Bot, Bookmark, Heart, LayoutGrid,
  Building2, FlaskConical, Bell, History, ListVideo, Briefcase, Settings,
  ChevronLeft, Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";

const items = [
  { icon: LayoutDashboard, label: "Dashboard", id: "dashboard", panel: "chart" },
  { icon: Radar, label: "Live Scanner", id: "scanner", panel: "scanner" },
  { icon: Puzzle, label: "Pattern Builder", id: "builder" },
  { icon: Bot, label: "AI Scanner", id: "ai-scanner", panel: "insights" },
  { icon: Bookmark, label: "Saved Patterns", id: "saved" },
  { icon: Heart, label: "Favorites", id: "favorites" },
  { icon: LayoutGrid, label: "Heatmap", id: "heatmap" },
  { icon: Building2, label: "Sector Analysis", id: "sectors" },
  { icon: FlaskConical, label: "Backtesting", id: "backtest" },
  { icon: Bell, label: "Alerts", id: "alerts" },
  { icon: History, label: "History", id: "history" },
  { icon: ListVideo, label: "Watchlist", id: "watchlist" },
  { icon: Briefcase, label: "Portfolio", id: "portfolio" },
  { icon: Settings, label: "Settings", id: "settings" },
];

export default function Sidebar() {
  const { sidebarCollapsed, toggleSidebar, activeSection, setActiveSection, showPanel, showToast, scannerData } = useStore();
  const patternCount = scannerData.filter((r) => r.alerted).length;

  const getBadge = (id: string) => {
    if (id === "scanner" && patternCount > 0) return patternCount;
    return undefined;
  };

  const handleNav = (id: string, panel?: string) => {
    setActiveSection(id);

    // Map sidebar items to actions
    switch (id) {
      case "scanner":
        showPanel("scanner");
        break;
      case "dashboard":
        showPanel("chart");
        showPanel("insights");
        break;
      case "ai-scanner":
        showPanel("insights");
        break;
      case "breadth":
        showPanel("breadth");
        break;
      case "heatmap":
        showToast("Heatmap — coming soon");
        break;
      case "builder":
        showPanel("builder");
        break;
      case "saved":
        showToast("Saved Patterns — coming soon");
        break;
      case "favorites":
        showToast("Favorites — coming soon");
        break;
      case "sectors":
        showToast("Sector Analysis — coming soon");
        break;
      case "backtest":
        showToast("Backtesting — coming soon");
        break;
      case "alerts":
        showToast("Alerts — coming soon");
        break;
      case "history":
        showToast("History — coming soon");
        break;
      case "watchlist":
        showToast("Watchlist — coming soon");
        break;
      case "portfolio":
        showToast("Portfolio — coming soon");
        break;
      case "settings":
        showToast("Settings — coming soon");
        break;
    }
  };

  return (
    <aside
      className="h-[calc(100vh-64px)] border-r flex flex-col transition-all duration-300 relative bg-[#0B0F17]"
      style={{
        width: sidebarCollapsed ? 56 : 220,
        borderColor: "rgba(255,255,255,0.08)",
      }}
    >
      <button
        onClick={toggleSidebar}
        className="absolute -right-3 top-3 w-6 h-6 rounded-full glass flex items-center justify-center z-10 hover:bg-white/10 transition-colors"
      >
        <ChevronLeft size={12} className={cn("transition-transform", sidebarCollapsed && "rotate-180")} />
      </button>

      <div className="flex-1 overflow-y-auto py-2 px-2 space-y-0.5">
        {items.map((item) => {
          const Icon = item.icon;
          const badge = getBadge(item.id);
          const isActive = activeSection === item.id;
          return (
            <button
              key={item.id}
              onClick={() => handleNav(item.id, item.panel)}
              className={cn(
                "w-full flex items-center gap-2.5 rounded-lg text-xs font-medium transition-all duration-150 group relative",
                sidebarCollapsed ? "justify-center p-2.5" : "px-3 py-2.5",
                isActive
                  ? "bg-[#3B82F6]/10 text-[#3B82F6]"
                  : "text-[#94A3B8] hover:text-white hover:bg-white/5"
              )}
            >
              <Icon size={16} className="shrink-0" />
              {!sidebarCollapsed && (
                <>
                  <span className="truncate">{item.label}</span>
                  {badge !== undefined && (
                    <span className="ml-auto text-[10px] font-bold px-1.5 py-0.5 rounded-full bg-[#EF4444]/15 text-[#EF4444]">
                      {badge}
                    </span>
                  )}
                </>
              )}
              {sidebarCollapsed && badge !== undefined && (
                <span className="absolute -top-0.5 -right-0.5 w-3.5 h-3.5 rounded-full bg-[#EF4444] text-[8px] font-bold flex items-center justify-center">
                  {badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      <div className={cn("p-2 border-t", sidebarCollapsed && "flex justify-center")}
        style={{ borderColor: "rgba(255,255,255,0.08)" }}>
        <div className={cn("panel p-3", sidebarCollapsed ? "p-2" : "")}>
          {!sidebarCollapsed ? (
            <div className="flex items-center gap-2">
              <Sparkles size={14} className="text-[#8B5CF6]" />
              <div className="flex-1 min-w-0">
                <div className="text-xs font-medium">AI Scanner</div>
                <div className="text-[10px] text-[#94A3B8]">{patternCount > 0 ? `${patternCount} patterns found` : "Ready"}</div>
              </div>
            </div>
          ) : (
            <Sparkles size={16} className="text-[#8B5CF6]" />
          )}
        </div>
      </div>
    </aside>
  );
}
