"""
BCC Dashboard — Streamlit app for Big Candle + Consolidation Scanner
Run: streamlit run bcc_dashboard.py
"""
import streamlit as st
import pandas as pd, numpy as np, os, glob, hashlib
from datetime import datetime, timezone
import altair as alt

st.set_page_config(page_title="BCC Scanner", page_icon="📊", layout="wide")

DATA_DIR = "C:/Users/pc/Downloads/stock hist data/comprehensive_data"
OUT_DIR = "C:/Users/pc/Downloads/stock hist data/backtest_results"
SEEN_FILE = os.path.join(OUT_DIR, "seen_patterns_bcc.csv")

# ─── DEFAULT PARAMS ───
DEFAULT = {
    "body_mult": 2.0,
    "vol_mult": 1.5,
    "wick_pct": 0.20,
    "avg_period": 20,
    "consol_min": 3,
    "consol_body_pct": 0.30,
    "consol_max_range_pct": 0.05,
}


# ─── LOAD DATA ───
@st.cache_data(show_spinner="Loading stock list...")
def load_stock_list():
    files = sorted(glob.glob(f"{DATA_DIR}/*_ONE_DAY.csv"))
    return [os.path.basename(f).replace("_ONE_DAY.csv", "") for f in files]


@st.cache_data(show_spinner="Loading seen patterns...")
def load_seen():
    if not os.path.exists(SEEN_FILE):
        return pd.DataFrame(columns=[
            "pattern_id", "symbol", "trigger_date", "trigger_type", "trigger_close",
            "trigger_body", "body_vs_avg", "vol_vs_avg", "wick_pct", "rsi",
            "consol_candles", "consol_range_pct", "consol_vol_vs_trigger",
            "status", "last_close", "change_pct", "detected_date"
        ])
    df = pd.read_csv(SEEN_FILE)
    if "trigger_date" in df.columns:
        df["trigger_date"] = pd.to_datetime(df["trigger_date"])
    if "detected_date" in df.columns:
        df["detected_date"] = pd.to_datetime(df["detected_date"])
    return df


@st.cache_data(show_spinner="Loading stock data...")
def load_stock_csv(symbol):
    f = f"{DATA_DIR}/{symbol}_ONE_DAY.csv"
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


# ─── SCAN FUNCTION ───
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g = gain.rolling(period).mean()
    avg_l = loss.rolling(period).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def scan_for_pattern(df, symbol, params, seen_set):
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    avg_p = params["avg_period"]
    df["avg_body"] = df["body"].rolling(avg_p, min_periods=avg_p).mean().shift(1)
    df["avg_vol"] = df["volume"].rolling(avg_p, min_periods=avg_p).mean().shift(1)
    df["rsi_val"] = rsi(df["close"], 14).shift(1)
    new = []
    for i in range(avg_p, n):
        row = df.iloc[i]
        ab = row["avg_body"]
        av = row["avg_vol"]
        if pd.isna(ab) or pd.isna(av) or ab == 0 or av == 0:
            continue
        body = row["body"]
        tr = row["range"]
        uw = row["high"] - max(row["close"], row["open"])
        if tr == 0:
            continue
        wr = uw / tr
        if not (body > ab * params["body_mult"] and row["volume"] > av * params["vol_mult"] and wr < params["wick_pct"]):
            continue
        ttype = "BULLISH" if row["close"] > row["open"] else "BEARISH"
        tclose = row["close"]
        tdate = pd.Timestamp(row["datetime"]).strftime("%Y-%m-%d")
        raw_id = f"{symbol}_{tdate}"
        pat_id = hashlib.md5(raw_id.encode()).hexdigest()[:12]
        if pat_id in seen_set:
            continue
        cc = 0
        ch = row["high"]
        cl = row["low"]
        ce = -1
        cvols = []
        for j in range(i + 1, n):
            c = df.iloc[j]
            ch = max(ch, c["high"])
            cl = min(cl, c["low"])
            cr_pct = (ch - cl) / tclose
            if cr_pct > params["consol_max_range_pct"]:
                if cc >= params["consol_min"]:
                    ce = j - 1
                    break
                cc = 0
                ch = c["high"]
                cl = c["low"]
                cvols = []
                continue
            cb = abs(c["close"] - c["open"])
            crr = c["high"] - c["low"]
            is_small = (cb / crr < params["consol_body_pct"]) if crr > 0 else True
            if is_small:
                cc += 1
                cvols.append(c["volume"])
            else:
                if cc >= params["consol_min"]:
                    ce = j - 1
                    break
                cc = 0
                ch = c["high"]
                cl = c["low"]
                cvols = []
        if cc >= params["consol_min"] and ce < 0:
            ce = min(j, n - 1)
        if ce < 0:
            continue
        lc = df.iloc[ce]["close"]
        chg = (lc - tclose) / tclose * 100
        if chg > 2:
            status = "BROKEN UP"
        elif chg < -2:
            status = "BROKEN DOWN"
        else:
            status = "CONSOLIDATING"
        acv = np.mean(cvols) if cvols else 0
        rv = row["rsi_val"]
        rs = f"{rv:.0f}" if not pd.isna(rv) else "N/A"
        new.append({
            "pattern_id": pat_id, "symbol": symbol,
            "trigger_date": tdate, "trigger_type": ttype,
            "trigger_close": round(tclose, 2), "trigger_body": round(body, 2),
            "body_vs_avg": round(body / ab, 2), "vol_vs_avg": round(row["volume"] / av, 2),
            "wick_pct": round(wr * 100, 1), "rsi": rs,
            "consol_candles": cc, "consol_range_pct": round((ch - cl) / tclose * 100, 2),
            "consol_vol_vs_trigger": round(acv / row["volume"], 2) if row["volume"] > 0 else 0,
            "status": status, "last_close": round(lc, 2), "change_pct": round(chg, 2),
            "detected_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return new


# ─── UI ───
st.title("📊 BCC Scanner Dashboard")
st.caption("Big Candle + Consolidation Pattern Scanner — 493 Indian Stocks")

stocks = load_stock_list()
seen = load_seen()

# ─── SIDEBAR ───
with st.sidebar:
    st.header("⚙️ Parameters")
    body_mult = st.slider("Big candle body multiplier", 1.5, 5.0, DEFAULT["body_mult"], 0.1)
    vol_mult = st.slider("Volume multiplier", 1.0, 5.0, DEFAULT["vol_mult"], 0.1)
    wick_pct = st.slider("Max upper wick %", 5, 50, int(DEFAULT["wick_pct"] * 100)) / 100
    consol_min = st.slider("Min consolidation candles", 2, 10, DEFAULT["consol_min"])
    consol_max_range = st.slider("Max consolidation range %", 1, 10, int(DEFAULT["consol_max_range_pct"] * 100)) / 100

    params = {
        "body_mult": body_mult, "vol_mult": vol_mult, "wick_pct": wick_pct,
        "avg_period": 20, "consol_min": consol_min,
        "consol_body_pct": 0.30, "consol_max_range_pct": consol_max_range,
    }

    st.divider()
    st.metric("Stocks", len(stocks))
    st.metric("Patterns Found", len(seen))
    if len(seen) > 0:
        st.metric("Active (Consolidating)", len(seen[seen["status"] == "CONSOLIDATING"]))

    st.divider()
    with st.expander("📄 CSV Export"):
        if st.button("Export Seen Patterns"):
            csv = seen.to_csv(index=False)
            st.download_button("Download CSV", csv, "bcc_patterns.csv", "text/csv")

# ─── TABS ───
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "📋 Patterns", "🔍 Stock View", "🖥️ Scanner", "📈 Stock List"
])

# ─── TAB 1: OVERVIEW ───
with tab1:
    st.subheader("Pattern Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patterns", len(seen))
    with col2:
        bull_count = len(seen[seen["trigger_type"] == "BULLISH"]) if len(seen) > 0 else 0
        st.metric("Bullish", bull_count)
    with col3:
        bear_count = len(seen[seen["trigger_type"] == "BEARISH"]) if len(seen) > 0 else 0
        st.metric("Bearish", bear_count)
    with col4:
        active = len(seen[seen["status"] == "CONSOLIDATING"]) if len(seen) > 0 else 0
        st.metric("Active (Consolidating)", active)

    if len(seen) > 0:
        st.divider()
        cola, colb = st.columns(2)

        with cola:
            st.subheader("Status Distribution")
            status_counts = seen["status"].value_counts().reset_index()
            status_counts.columns = ["status", "count"]
            c = alt.Chart(status_counts).mark_arc().encode(
                theta="count", color="status", tooltip=["status", "count"]
            ).properties(height=300)
            st.altair_chart(c, use_container_width=True)

        with colb:
            st.subheader("Type Distribution")
            type_counts = seen["trigger_type"].value_counts().reset_index()
            type_counts.columns = ["type", "count"]
            c = alt.Chart(type_counts).mark_arc().encode(
                theta="count", color="type", tooltip=["type", "count"]
            ).properties(height=300)
            st.altair_chart(c, use_container_width=True)

        st.divider()
        st.subheader("Top Stocks by Pattern Count")
        top_stocks = seen["symbol"].value_counts().head(20).reset_index()
        top_stocks.columns = ["symbol", "count"]
        c = alt.Chart(top_stocks).mark_bar().encode(
            x="count", y=alt.Y("symbol", sort="-x"), tooltip=["symbol", "count"]
        ).properties(height=500)
        st.altair_chart(c, use_container_width=True)

        st.divider()
        st.subheader("Patterns Over Time")
        if "trigger_date" in seen.columns:
            date_counts = seen.groupby(seen["trigger_date"].dt.to_period("M")).size().reset_index(name="count")
            date_counts["trigger_date"] = date_counts["trigger_date"].astype(str)
            c = alt.Chart(date_counts).mark_line(point=True).encode(
                x="trigger_date:T", y="count", tooltip=["trigger_date", "count"]
            ).properties(height=300)
            st.altair_chart(c, use_container_width=True)

# ─── TAB 2: PATTERNS ───
with tab2:
    st.subheader("Pattern Explorer")
    if len(seen) > 0:
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        df = seen.copy()

        with fcol1:
            search = st.text_input("🔍 Search symbol", "")
        with fcol2:
            type_filter = st.selectbox("Type", ["All", "BULLISH", "BEARISH"])
        with fcol3:
            status_filter = st.selectbox("Status", ["All", "CONSOLIDATING", "BROKEN UP", "BROKEN DOWN"])
        with fcol4:
            date_range = st.date_input("From date", value=None)

        if search:
            df = df[df["symbol"].str.contains(search.upper(), na=False)]
        if type_filter != "All":
            df = df[df["trigger_type"] == type_filter]
        if status_filter != "All":
            df = df[df["status"] == status_filter]

        df_display = df.sort_values("trigger_date", ascending=False).head(500)
        cols = ["symbol", "trigger_date", "trigger_type", "trigger_close",
                "body_vs_avg", "vol_vs_avg", "rsi", "consol_candles",
                "consol_range_pct", "status", "change_pct"]
        cols = [c for c in cols if c in df_display.columns]
        st.dataframe(df_display[cols], use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(df_display)} of {len(seen)} patterns")
    else:
        st.info("No patterns found yet. Run the scanner first.")

# ─── TAB 3: STOCK VIEW ───
with tab3:
    st.subheader("Individual Stock Analysis")
    selected = st.selectbox("Select a stock", stocks, index=stocks.index("RELIANCE") if "RELIANCE" in stocks else 0)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 Scan This Stock Now"):
            with st.spinner(f"Scanning {selected}..."):
                df = load_stock_csv(selected)
                if df is not None:
                    seen_set = set(seen["pattern_id"].tolist()) if len(seen) > 0 else set()
                    pats = scan_for_pattern(df, selected, params, seen_set)
                    if pats:
                        st.success(f"Found {len(pats)} new patterns!")
                        for p in pats:
                            st.code(
                                f"[{p['trigger_type']:>8s}] @ {p['trigger_close']:>10.2f} | "
                                f"{p['trigger_date']} | Body:{p['body_vs_avg']:.1f}x Vol:{p['vol_vs_avg']:.1f}x | "
                                f"Consol:{p['consol_candles']}d ({p['consol_range_pct']:.1f}%) | {p['status']}"
                            )
                    else:
                        st.info("No new patterns found.")
                else:
                    st.error("Stock data not found.")

    with col2:
        st.caption(f"Data: {selected}")
        if len(seen) > 0:
            stock_pats = seen[seen["symbol"] == selected]
            if len(stock_pats) > 0:
                st.metric("Historical Patterns", len(stock_pats))
                consol = len(stock_pats[stock_pats["status"] == "CONSOLIDATING"])
                up = len(stock_pats[stock_pats["status"] == "BROKEN UP"])
                down = len(stock_pats[stock_pats["status"] == "BROKEN DOWN"])
                st.caption(f"Consolidating: {consol} | Broken Up: {up} | Broken Down: {down}")

    st.divider()
    df = load_stock_csv(selected)
    if df is not None:
        st.subheader(f"{selected} — Price Chart (Last 500 days)")
        chart_df = df.tail(500).copy()
        base = alt.Chart(chart_df).encode(x="datetime:T")
        bars = base.mark_bar().encode(
            alt.Y("close:Q").title("Price"),
            color=alt.condition(
                alt.datum.close >= alt.datum.open,
                alt.value("#26a69a"),
                alt.value("#ef5350")
            )
        )
        st.altair_chart(bars.interactive(), use_container_width=True)

        with st.expander("📋 Raw Data (Last 50)"):
            st.dataframe(df.tail(50)[["datetime", "open", "high", "low", "close", "volume"]],
                        use_container_width=True, hide_index=True)

# ─── TAB 4: SCANNER ───
with tab4:
    st.subheader("Run Scanner")
    st.markdown(f"""
    **Parameters:**
    - Big candle body > {params['body_mult']}x avg
    - Volume > {params['vol_mult']}x avg
    - Upper wick < {params['wick_pct']*100:.0f}% of range
    - Consolidation: {params['consol_min']}+ candles, range < {params['consol_max_range_pct']*100:.0f}%
    """)

    st.caption(f"Previously seen: {len(seen)} patterns")

    scan_all = st.button("▶️ Scan ALL 493 Stocks")

    if scan_all:
        seen_set = set(seen["pattern_id"].tolist()) if len(seen) > 0 else set()
        all_new = []
        progress = st.progress(0)
        status_text = st.empty()
        files_list = sorted(glob.glob(f"{DATA_DIR}/*_ONE_DAY.csv"))
        for idx, f in enumerate(files_list):
            sym = os.path.basename(f).replace("_ONE_DAY.csv", "")
            df = pd.read_csv(f)
            df["datetime"] = pd.to_datetime(df["datetime"])
            pats = scan_for_pattern(df, sym, params, seen_set)
            all_new.extend(pats)
            progress.progress((idx + 1) / len(files_list))
            if (idx + 1) % 50 == 0:
                status_text.text(f"Scanned {idx+1}/{len(files_list)}... ({len(all_new)} new found)")

        progress.empty()
        status_text.empty()

        if all_new:
            new_df = pd.DataFrame(all_new)
            new_df.to_csv(SEEN_FILE, mode="a", header=not os.path.exists(SEEN_FILE), index=False)
            st.success(f"✅ Found {len(all_new)} new patterns!")
            for p in all_new[:20]:
                st.code(
                    f"[{p['symbol']:16s}] {p['trigger_type']:>8s} @ {p['trigger_close']:>10.2f} | "
                    f"{p['trigger_date']} | Body:{p['body_vs_avg']:.1f}x Vol:{p['vol_vs_avg']:.1f}x | "
                    f"Consol:{p['consol_candles']}d ({p['consol_range_pct']:.1f}%) | {p['status']}"
                )
            if len(all_new) > 20:
                st.caption(f"... and {len(all_new)-20} more")
            st.cache_data.clear()
            st.rerun()
        else:
            st.info("No new patterns found.")

    st.divider()
    st.subheader("Scan Specific Stocks")
    scan_stocks = st.multiselect("Select stocks", stocks, default=[])
    if scan_stocks and st.button("▶️ Scan Selected"):
        seen_set = set(seen["pattern_id"].tolist()) if len(seen) > 0 else set()
        all_new = []
        for sym in scan_stocks:
            df = load_stock_csv(sym)
            if df is not None:
                pats = scan_for_pattern(df, sym, params, seen_set)
                all_new.extend(pats)
        if all_new:
            new_df = pd.DataFrame(all_new)
            new_df.to_csv(SEEN_FILE, mode="a", header=not os.path.exists(SEEN_FILE), index=False)
            st.success(f"Found {len(all_new)} new patterns!")
            for p in all_new:
                st.code(
                    f"[{p['symbol']:16s}] {p['trigger_type']:>8s} @ {p['trigger_close']:>10.2f} | "
                    f"{p['trigger_date']} | Body:{p['body_vs_avg']:.1f}x Vol:{p['vol_vs_avg']:.1f}x | "
                    f"Consol:{p['consol_candles']}d ({p['consol_range_pct']:.1f}%) | {p['status']}"
                )
            st.cache_data.clear()
        else:
            st.info("No new patterns found.")

# ─── TAB 5: STOCK LIST ───
with tab5:
    st.subheader(f"All Stocks ({len(stocks)})")
    search_stock = st.text_input("Search", "", placeholder="Type to filter...")
    filtered = [s for s in stocks if search_stock.upper() in s] if search_stock else stocks
    cols_per_row = 5
    for i in range(0, len(filtered), cols_per_row):
        row_stocks = filtered[i:i+cols_per_row]
        cols = st.columns(cols_per_row)
        for j, s in enumerate(row_stocks):
            with cols[j]:
                pat_count = len(seen[seen["symbol"] == s]) if len(seen) > 0 else 0
                if pat_count > 0:
                    st.button(f"📊 {s[:12]:12s} ({pat_count})", key=f"stock_{s}",
                              on_click=lambda sym=s: st.session_state.update({"sel_stock": sym}),
                              use_container_width=True)
                else:
                    st.button(f"  {s[:12]:12s}", key=f"stock_{s}",
                              on_click=lambda sym=s: st.session_state.update({"sel_stock": sym}),
                              use_container_width=True)
    st.caption(f"Showing {len(filtered)} stocks")

# Handle stock selection from tab5
if "sel_stock" in st.session_state and st.session_state.sel_stock:
    sel = st.session_state.sel_stock
    st.session_state.sel_stock = None
    with tab3:
        sel_idx = stocks.index(sel) if sel in stocks else 0
        st.selectbox("Selected", stocks, index=sel_idx, key="stock_selector")
        st.rerun()
