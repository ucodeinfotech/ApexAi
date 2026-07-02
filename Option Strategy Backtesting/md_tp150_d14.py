"""MD_TP150_D14: NIFTY50 CALL multi-day hold, TP=150pts, max 14 days (corrected engine)."""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WORK_DIR = Path(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
LOT = 50
DB_PATH = Path(r"C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb")
CUT_TIME = pd.Timestamp("14:15").time()
REQUIRED_SPOT_COLUMNS = {"datetime", "open", "high", "low", "close", "volume"}
REQUIRED_OPTION_COLUMNS = {"timestamp", "close", "strike", "option_type", "expiry_code", "expiry_flag", "atm_distance"}


def validate_spot_dataframe(name: str, df: pd.DataFrame) -> None:
    missing = REQUIRED_SPOT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")
    required_cols = list(REQUIRED_SPOT_COLUMNS)
    if df[required_cols].isna().any().any():
        raise ValueError(f"{name} contains missing values in required columns")
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        raise ValueError(f"{name}.datetime column must be datetime type")


def validate_spot_files(base_dir: Path) -> None:
    if not base_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {base_dir}")
    for name in ("NIFTY50_ONE_HOUR.csv", "NIFTY50_FIVE_MINUTE.csv"):
        path = base_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Required spot file is missing: {path}")


def validate_database(db_path: Path) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {db_path}")
    with duckdb.connect(str(db_path)) as con:
        tables = set(r[0] for r in con.execute("SHOW TABLES").fetchall())
        if "options_data_clean" not in tables:
            raise ValueError("DuckDB does not contain table 'options_data_clean'")
        cols = set(r[0] for r in con.execute("DESCRIBE options_data_clean").fetchall())
        missing = REQUIRED_OPTION_COLUMNS - cols
        if missing:
            raise ValueError(f"options_data_clean is missing required columns: {sorted(missing)}")


def load_spot_data(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_spot_files(base_dir)
    h1 = pd.read_csv(base_dir / "NIFTY50_ONE_HOUR.csv", parse_dates=["datetime"])
    m5 = pd.read_csv(base_dir / "NIFTY50_FIVE_MINUTE.csv", parse_dates=["datetime"])
    validate_spot_dataframe("NIFTY50_ONE_HOUR.csv", h1)
    validate_spot_dataframe("NIFTY50_FIVE_MINUTE.csv", m5)
    for df in (h1, m5):
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return h1, m5


def _datetime_to_int64(ts: pd.Timestamp) -> int:
    return np.datetime64(ts, "us").astype("int64")


def find_spot_entries(h1: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
    me = m5["datetime"].astype("int64").values
    b = (h1["close"] - h1["open"]).abs()
    g = h1["close"] > h1["open"]
    rr = h1["close"] < h1["open"]
    m5_times = m5["datetime"].dt.time.values
    trades = []

    for i in range(1, len(h1)):
        if not (rr.iloc[i - 1] and g.iloc[i]):
            continue
        if h1["open"].iloc[i] > h1["close"].iloc[i - 1] or h1["close"].iloc[i] < h1["open"].iloc[i - 1]:
            continue
        if b.iloc[i] < b.iloc[i - 1] * 0.5 or h1["datetime"].iloc[i].hour == 9:
            continue

        lv = h1["high"].iloc[i]
        ts = h1["datetime"].iloc[i]
        idx = np.searchsorted(me, _datetime_to_int64(ts), side="right")
        if idx >= len(m5):
            continue

        bi = idx
        while bi < len(m5) and m5["close"].iloc[bi] <= lv:
            bi += 1
        if bi >= len(m5) - 1:
            continue

        ri = bi + 1
        while ri < len(m5):
            if m5["low"].iloc[ri] < lv and m5["close"].iloc[ri] > lv and m5_times[ri] < CUT_TIME:
                break
            ri += 1
        if ri >= len(m5):
            continue

        entry_price = m5["close"].iloc[ri]
        if entry_price - m5["low"].iloc[ri] <= 0:
            continue

        he = entry_price
        for j in range(ri, len(m5)):
            ca = m5["high"].iloc[j] - m5["low"].iloc[j]
            if m5["high"].iloc[j] > he:
                he = m5["high"].iloc[j]
            if m5["close"].iloc[j] < he - 55 * ca:
                trades.append({"entry_dt": m5["datetime"].iloc[ri], "yr": ts.year, "mo": ts.month})
                break

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df

    trades_df["ed_naive"] = trades_df["entry_dt"].dt.tz_localize(None)
    return trades_df


def load_atm_data(db_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_database(db_path)
    with duckdb.connect(str(db_path)) as con:
        df_atm = con.execute(
            """SELECT timestamp, close, strike FROM options_data_clean
               WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND atm_distance=0
               ORDER BY timestamp"""
        ).fetchdf()

    if df_atm.empty:
        raise ValueError("No ATM option data found in the database for the selected filter")
    df_atm["timestamp"] = pd.to_datetime(df_atm["timestamp"], utc=False)
    if df_atm["timestamp"].isna().any():
        raise ValueError("ATM option timestamps contain invalid or missing values")
    return (
        df_atm["timestamp"].values.astype("datetime64[us]"),
        df_atm["close"].values.astype(float),
        df_atm["strike"].values.astype(float),
    )


def lookup_atm(atm_ts: np.ndarray, atm_cl: np.ndarray, atm_st: np.ndarray, ed: pd.Timestamp) -> tuple[int, float, float]:
    ts64 = np.datetime64(ed, "us")
    i = np.searchsorted(atm_ts, ts64)
    if i >= len(atm_ts):
        return len(atm_ts) - 1, atm_cl[-1], atm_st[-1]
    if i == 0:
        return 0, atm_cl[0], atm_st[0]
    return (i, atm_cl[i], atm_st[i]) if atm_ts[i] == ts64 else (i - 1, atm_cl[i - 1], atm_st[i - 1])


def build_strike_cache(db_path: Path, strikes: list[int]) -> dict[int, dict[str, np.ndarray]]:
    if not strikes:
        return {}

    strike_list = ",".join(map(str, strikes))
    with duckdb.connect(str(db_path)) as con:
        df_all = con.execute(
            f"""SELECT timestamp, close, strike FROM options_data_clean
                WHERE option_type='CALL' AND expiry_code=1 AND expiry_flag='WEEK' AND strike IN ({strike_list})
                ORDER BY strike, timestamp"""
        ).fetchdf()

    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=False)
    strike_cache = {}
    for stk, grp in df_all.groupby("strike"):
        strike_cache[int(stk)] = {
            "ts": grp["timestamp"].values.astype("datetime64[us]"),
            "cl": grp["close"].values.astype(float),
        }
    return strike_cache


def build_trade_infos(trades: pd.DataFrame, atm_ts: np.ndarray, atm_cl: np.ndarray, atm_st: np.ndarray, strike_cache: dict[int, dict[str, np.ndarray]]) -> list[dict | None]:
    trade_infos = []
    for index, row in trades.iterrows():
        entry = row["ed_naive"]
        ts64 = np.datetime64(entry, "us")
        i = np.searchsorted(atm_ts, ts64)
        si = (
            len(atm_ts) - 1
            if i >= len(atm_ts)
            else 0
            if i == 0
            else i if atm_ts[i] == ts64
            else i - 1
        )
        st = int(atm_st[si])
        sd = strike_cache.get(st)
        if sd is None:
            trade_infos.append(None)
            continue

        s_idx = np.searchsorted(sd["ts"], atm_ts[si])
        if s_idx >= len(sd["cl"]):
            trade_infos.append(None)
            continue

        trade_infos.append(
            {
                "strike": st,
                "ep": float(sd["cl"][s_idx]),
                "s_idx": int(s_idx),
                "stk_data": sd,
                "yr": int(row["yr"]),
                "mo": int(row["mo"]),
            }
        )

    return trade_infos


def run_strategy(trade_infos: list[dict | None], tp: float, maxd: int) -> tuple[np.ndarray, list[int], list[int]]:
    pnls, yrs, mos = [], [], []
    max_ns = int(maxd * 86_400 * 1_000_000)

    for info in trade_infos:
        if info is None:
            continue

        sd = info["stk_data"]
        s_idx = info["s_idx"]
        ep = info["ep"]
        end_ns = sd["ts"][s_idx] + np.timedelta64(max_ns, "us")
        result = None

        for i in range(s_idx + 1, min(s_idx + 3_000, len(sd["cl"]))):
            if sd["ts"][i] > end_ns:
                result = sd["cl"][i] - ep
                break
            if sd["cl"][i] - ep >= tp:
                result = sd["cl"][i] - ep
                break

        if result is None:
            continue

        pnls.append(round(result, 1))
        yrs.append(info["yr"])
        mos.append(info["mo"])

    return np.array(pnls), yrs, mos


def print_summary(pnls: np.ndarray, yrs: list[int], mos: list[int], tp: float, maxd: int) -> None:
    n = len(pnls)
    net = pnls.sum() if n else 0.0

    if n == 0:
        print("No trades were generated for the selected strategy and data window.")
        return

    wr = float((pnls > 0).mean() * 100)
    avg = float(pnls.mean())
    std = float(pnls.std()) if n > 1 else 0.0
    sharpe = float(avg / std * np.sqrt(252)) if std > 0 else 0.0
    cum = np.cumsum(pnls)
    mx = np.maximum.accumulate(cum)
    mdd = float((mx - cum).max()) if n else 0.0
    calmar = float(net / mdd) if mdd > 0 else 0.0
    pf = float(pnls[pnls > 0].sum() / abs(pnls[pnls < 0].sum())) if (pnls < 0).sum() > 0 else 999.0
    wl = float(pnls[pnls > 0].mean() / abs(pnls[pnls < 0].mean())) if (pnls < 0).sum() > 0 else 999.0

    print(f"\n{'=' * 55}")
    print(f"MD_TP150_D14  |  TP={tp}  |  MaxD={maxd}  |  LOT={LOT}")
    print(f"{'=' * 55}")
    print(f"Total trades:    {n}")
    print(f"Net PnL:         {net:>+8,.0f} pts  (Rs {net * LOT:>+,.0f})")
    print(f"Win Rate:        {wr:.1f}%")
    print(f"Avg PnL:         {avg:+.1f} pts")
    print(f"Std PnL:         {std:.1f} pts")
    print(f"Sharpe:          {sharpe:.2f}")
    print(f"Calmar:          {calmar:.1f}x")
    print(f"Profit Factor:   {pf:.2f}x")
    print(f"W/L Ratio:       {wl:.2f}x")
    print(f"Max Drawdown:    {mdd:>+8,.0f} pts  (Rs {mdd * LOT:>+,.0f})")
    print(f"Best Trade:      {pnls.max():+8.0f} pts")
    print(f"Worst Trade:     {pnls.min():+8.0f} pts")
    print(f"Win trades:      {(pnls > 0).sum()}")
    print(f"Loss trades:     {(pnls < 0).sum()}")

    print(f"\n{'=' * 55}")
    print("Yearly Breakdown")
    print(f"{'=' * 55}")
    print(f"{'Year':<6} {'Trades':>7} {'Net Pts':>10} {'Net Rs':>12} {'WR':>6} {'Avg':>7} {'Sharpe':>7}")
    print("-" * 55)
    yr_data: dict[int, list[float]] = {}
    for i, yr in enumerate(yrs):
        yr_data.setdefault(yr, []).append(pnls[i])
    for yr in sorted(yr_data):
        p = np.array(yr_data[yr])
        s_n = len(p)
        s_net = float(p.sum())
        s_wr = float((p > 0).mean() * 100)
        s_avg = float(p.mean())
        s_std = float(p.std()) if s_n > 1 else 0.0
        s_sh = float(s_avg / s_std * np.sqrt(252)) if s_std > 0 else 0.0
        print(
            f"{yr:<6} {s_n:>7} {s_net:>+9,.0f} Rs{s_net * LOT:>+9,.0f} {s_wr:>5.1f}% {s_avg:>+7.1f} {s_sh:>6.2f}"
        )
    print(
        f"{'TOTAL':<6} {n:>7} {net:>+9,.0f} Rs{net * LOT:>+9,.0f} {wr:>5.1f}% {avg:>+7.1f} {sharpe:>6.2f}"
    )

    print(f"\n{'=' * 55}")
    print("Monthly Breakdown")
    print(f"{'=' * 55}")
    print(f"{'Month':<8} {'Trades':>7} {'Net Pts':>10} {'Net Rs':>12} {'WR':>6} {'Avg':>7}")
    print("-" * 50)
    mo_data: dict[int, list[float]] = {}
    for i, mo in enumerate(mos):
        mo_data.setdefault(mo, []).append(pnls[i])
    mo_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for mo in range(1, 13):
        p = np.array(mo_data.get(mo, []))
        if p.size == 0:
            continue
        s_n = len(p)
        s_net = float(p.sum())
        s_wr = float((p > 0).mean() * 100)
        s_avg = float(p.mean())
        print(
            f"{mo_names[mo - 1]:<8} {s_n:>7} {s_net:>+9,.0f} Rs{s_net * LOT:>+9,.0f} {s_wr:>5.1f}% {s_avg:>7.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MD_TP150_D14 strategy backtest")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=WORK_DIR,
        help="Path to the folder containing the CSV spot files",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help="Path to the DuckDB options history database",
    )
    parser.add_argument("--tp", type=float, default=150.0, help="Target profit in points")
    parser.add_argument("--maxd", type=int, default=14, help="Maximum holding period in days")
    parser.add_argument("--check-only", action="store_true", help="Validate input data files and database, then exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_only:
        load_spot_data(args.data_dir)
        validate_database(args.db_path)
        print("Data validation passed.")
        return

    h1, m5 = load_spot_data(args.data_dir)
    trades = find_spot_entries(h1, m5)

    if trades.empty:
        print("No spot entries were generated.")
        return

    trades_pre = trades[trades["ed_naive"] >= pd.Timestamp("2021-06-14")].reset_index(drop=True)
    print(f"Spot entries (option-matchable): {len(trades_pre)}")

    atm_ts, atm_cl, atm_st = load_atm_data(args.db_path)
    strike_set = set()
    for ed in trades_pre["ed_naive"]:
        _, _, st = lookup_atm(atm_ts, atm_cl, atm_st, ed)
        strike_set.add(int(st))

    strike_cache = build_strike_cache(args.db_path, sorted(strike_set))
    trade_infos = build_trade_infos(trades_pre, atm_ts, atm_cl, atm_st, strike_cache)
    pnls, yrs, mos = run_strategy(trade_infos, tp=args.tp, maxd=args.maxd)
    print_summary(pnls, yrs, mos, tp=args.tp, maxd=args.maxd)


if __name__ == "__main__":
    main()

