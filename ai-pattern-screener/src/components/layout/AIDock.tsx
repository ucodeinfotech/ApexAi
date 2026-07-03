"use client";

import { useState } from "react";
import { Bot, Send, Sparkles, X, ChevronUp, ChevronDown } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const suggestions = [
  "Find Bull Flags with >85% similarity",
  "Show stocks like RELIANCE",
  "Explain current ADANIGREEN pattern",
  "Highest confidence patterns now",
];

export default function AIDock() {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");

  return (
    <div className="fixed bottom-10 right-5 z-50 flex flex-col items-end gap-2">
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className="panel w-80 overflow-hidden"
          >
            {/* Header */}
            <div className="panel-header" style={{ borderColor: "rgba(139,92,246,0.2)" }}>
              <div className="flex items-center gap-2">
                <Sparkles size={14} className="text-[#8B5CF6]" />
                <span className="text-sm font-medium">AI Assistant</span>
              </div>
              <button onClick={() => setOpen(false)} className="btn-ghost p-0.5">
                <X size={12} />
              </button>
            </div>

            {/* Suggestions */}
            <div className="p-3 space-y-1.5">
              <div className="text-[10px] text-[#4B5563] uppercase tracking-wider font-medium mb-2">Suggested</div>
              {suggestions.map((s) => (
                <button
                  key={s}
                  className="w-full text-left text-xs px-2.5 py-2 rounded-lg glass-hover text-[#94A3B8] hover:text-white transition-colors"
                  onClick={() => setInput(s)}
                >
                  {s}
                </button>
              ))}
            </div>

            {/* Messages (static demo) */}
            <div className="px-3 pb-3 space-y-2">
              <div className="flex gap-2">
                <div className="w-5 h-5 rounded-full bg-[#8B5CF6]/20 flex items-center justify-center shrink-0 mt-0.5">
                  <Bot size={10} className="text-[#8B5CF6]" />
                </div>
                <div className="text-xs text-[#94A3B8] bg-white/5 rounded-lg p-2.5 leading-relaxed">
                  Found <span className="text-[#22C55E]">12 Bull Flags</span> and <span className="text-[#EF4444]">5 Bear Flags</span> matching your criteria. Top match: <span className="text-white font-medium">RELIANCE</span> at 94.2% similarity.
                </div>
              </div>
            </div>

            {/* Input */}
            <div className="px-3 pb-3">
              <div className="flex items-center gap-2 input">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask AI anything..."
                  className="bg-transparent border-none text-xs text-white flex-1 outline-none placeholder:text-[#4B5563]"
                />
                <button className="btn-ghost p-0.5">
                  <Send size={12} className="text-[#3B82F6]" />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <button
        onClick={() => setOpen(!open)}
        className="w-10 h-10 rounded-full glass flex items-center justify-center hover:bg-white/10 transition-colors border-[#8B5CF6]/30"
      >
        <Bot size={18} className="text-[#8B5CF6]" />
      </button>
    </div>
  );
}
