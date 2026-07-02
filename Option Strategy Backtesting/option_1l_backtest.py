"""
Full rupee backtest: 1 lot NIFTY, 1L capital, best option strategy
Generates comprehensive PDF with trade book
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, warnings, bisect, calendar
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
LOT=50  # NIFTY lot size

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 11, 'axes.labelsize': 9,
    'figure.dpi': 120, 'savefig.dpi': 150, 'font.family': 'sans-serif'})

# === Build trades ===
def atr(m5):
    tr=pd.concat([m5["high"]-m5["low"],abs(m5["high"]-m5["close"].shift(1)),abs(m5["low"]-m5["close"].shift(1))],axis=1).max(axis=1)
    return tr.ewm(span=14,min_periods=14,adjust=False).mean()
print("Building trades...")
h1=pd.read_csv("NIFTY50_ONE_HOUR.csv",parse_dates=["datetime"])
m5=pd.read_csv("NIFTY50_FIVE_MINUTE.csv",parse_dates=["datetime"])
for d in [h1,m5]:
    d["datetime"]=pd.to_datetime(d["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
    d.sort_values("datetime",inplace=True);d.reset_index(drop=True,inplace=True)
a5=atr(m5); me=m5["datetime"].astype("int64").values
CUT=pd.Timestamp("14:15").time()
trades=[]; b=(h1["close"]-h1["open"]).abs(); g=h1["close"]>h1["open"]; rr=h1["close"]<h1["open"]
for i in range(1,len(h1)):
    if not (rr.iloc[i-1] and g.iloc[i]): continue
    if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
    if b.iloc[i]<b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour==9: continue
    lv=h1["high"].iloc[i]; ts=h1["datetime"].iloc[i]
    idx=np.searchsorted(me,ts.asm8.view("int64"),side="right")
    if idx>=len(m5["close"]): continue
    bi=idx
    while bi<len(m5["close"]) and m5["close"].iloc[bi]<=lv: bi+=1
    if bi>=len(m5["close"])-1: continue
    ri=bi+1
    while ri<len(m5["close"]):
        if m5["low"].iloc[ri]<lv and m5["close"].iloc[ri]>lv and pd.Series(m5["datetime"]).dt.time.iloc[ri]<CUT: break
        ri+=1
    if ri>=len(m5["close"]): continue
    ed=m5["datetime"].iloc[ri]; ep_=m5["close"].iloc[ri]
    if ep_-m5["low"].iloc[ri]<=0: continue
    he=ep_
    for j in range(ri,len(m5["close"])):
        ca=a5.iloc[j]
        if pd.isna(ca): continue
        if m5["high"].iloc[j]>he: he=m5["high"].iloc[j]
        if m5["close"].iloc[j]<he-55*ca:
            trades.append({"entry_dt":ed,"exit_dt":m5["datetime"].iloc[j],"yr":ts.year,"mo":ts.month,"weekday":ed.weekday()}); break
trades=pd.DataFrame(trades)
trades["ed_naive"]=trades["entry_dt"].dt.tz_localize(None)
trades["xd_naive"]=trades["exit_dt"].dt.tz_localize(None)
exp_w_l=[]
for _,r in trades.iterrows():
    d=r["ed_naive"].date()
    da=(3-d.weekday())%7; da=da if da>0 else 7
    exp_w_l.append((r["ed_naive"]+pd.Timedelta(days=da)).replace(hour=15,minute=25))
trades["exp_w"]=exp_w_l
trades_pre=trades[trades["ed_naive"]>=pd.Timestamp("2021-06-14")].reset_index(drop=True)
print(f"Trades: {len(trades_pre)}")

# === Load WEEKLY option data ===
print("Loading option data...")
con=duckdb.connect(r'C:\Users\pc\Desktop\ucode\project\storage\duckdb\options_history.duckdb')
df_opt=con.execute("SELECT timestamp,close FROM options_data WHERE option_type='CALL' AND expiry_code=1 AND atm_distance=0 AND expiry_flag='WEEK' ORDER BY timestamp").fetchdf()
con.close()
df_opt["timestamp"]=pd.to_datetime(df_opt["timestamp"],utc=False)
ts_arr=df_opt["timestamp"].values.astype("datetime64[us]")
cl_arr=df_opt["close"].values.astype(float)

def lookup(ts):
    i=np.searchsorted(ts_arr,np.datetime64(ts,"us"))
    if i<=0: return 0,cl_arr[0]
    if i>=len(ts_arr): return len(ts_arr)-1,cl_arr[-1]
    return i-1,cl_arr[i-1]

ep_vals=np.array([lookup(ed)[1] for ed in trades_pre["ed_naive"]])
si_vals=np.array([lookup(ed)[0] for ed in trades_pre["ed_naive"]])

def exit_tp_maxd(s_idx, tp, max_days):
    end_ns=ts_arr[s_idx]+np.timedelta64(int(max_days*86400*1e6),"us"); ep=cl_arr[s_idx]
    for i in range(s_idx+1, min(s_idx+3000, len(cl_arr))):
        if ts_arr[i]>end_ns: return cl_arr[i], ts_arr[i]
        if cl_arr[i]-ep>=tp: return cl_arr[i], ts_arr[i]
    return None, None

# === RUN BEST STRATEGY ===
TP_VAL=30; MAX_D=7; PREM_MAX=130
pnls=[]; indices=[]; trade_book=[]
for i in range(len(ep_vals)):
    if ep_vals[i]>PREM_MAX: continue
    xp,xts=exit_tp_maxd(si_vals[i],TP_VAL,MAX_D)
    if xp is None: continue
    pnl_pts=round(xp-ep_vals[i],1)
    pnl_rs=round(pnl_pts*LOT,2)
    trade_book.append({"#":len(trade_book)+1,
        "Entry":trades_pre.iloc[i]["ed_naive"],
        "Exit":pd.Timestamp(xts).tz_localize(None) if isinstance(xts,np.datetime64) else xts,
        "EntryPx":ep_vals[i],"ExitPx":xp,
        "Pts":pnl_pts,"Rs":pnl_rs,"CumRs":0,
        "Yr":int(trades_pre.iloc[i]["yr"]),"Mo":int(trades_pre.iloc[i]["mo"]),
        "Weekday":int(trades_pre.iloc[i]["weekday"])})
    pnls.append(pnl_pts)

tb=pd.DataFrame(trade_book)
tb["CumRs"]=tb["Rs"].cumsum()
INIT_CAP=100000.0
tb["MTM"]=INIT_CAP+tb["CumRs"]
tb["Return%"]=round(tb["CumRs"]/INIT_CAP*100,2)

print(f"Trades: {len(tb)}")
print(f"Total PnL: Rs {tb['CumRs'].iloc[-1]:+,.0f}")
print(f"Final Capital: Rs {tb['MTM'].iloc[-1]:,.0f}")

# === STATISTICS ===
rs_arr=tb["Rs"].values
net_rs=rs_arr.sum(); wr=(rs_arr>0).mean()
aw=rs_arr[rs_arr>0].mean() if (rs_arr>0).sum()>0 else 0
al=rs_arr[rs_arr<0].mean() if (rs_arr<0).sum()>0 else 0
wl=aw/abs(al) if al!=0 else 999
pf_=rs_arr[rs_arr>0].sum()/abs(rs_arr[rs_arr<0].sum()) if (rs_arr<0).sum()>0 else 999
cum_rs=np.cumsum(rs_arr); mx=np.maximum.accumulate(cum_rs); dd_rs=mx-cum_rs; mdd_rs=dd_rs.max()
calmar=net_rs/mdd_rs if mdd_rs>0 else 999
sharpe=rs_arr.mean()/rs_arr.std()*np.sqrt(252) if rs_arr.std()>0 else 0
cum_ret_pct=(INIT_CAP+cum_rs)/INIT_CAP-1
max_runup=cum_ret_pct.max()*100
max_dd_pct=(dd_rs/(INIT_CAP+mx)).max()*100
avg_rs=rs_arr.mean()
n_wins=(rs_arr>0).sum(); n_losses=(rs_arr<0).sum()
total_cost=ep_vals[tb.index].sum()*LOT if len(tb)>0 else 0
avg_prem=ep_vals[tb.index].mean() if len(tb)>0 else 0

# Consecutive wins/losses
win_streaks=[sum(1 for _ in g) for k,g in __import__('itertools').groupby(rs_arr>0) if k]
loss_streaks=[sum(1 for _ in g) for k,g in __import__('itertools').groupby(rs_arr<=0) if k]
max_consec_win=max(win_streaks) if win_streaks else 0
max_consec_loss=max(loss_streaks) if loss_streaks else 0

stats_text=f"""
=== STRATEGY STATISTICS (Rupees) ===
Instrument:       NIFTY ATM CALL Weekly (1 lot = {LOT} units)
Capital:          Rs {INIT_CAP:,.0f}
Strategy:         TP30_Max7d + Premium < {PREM_MAX}

Total Trades:     {len(tb):,}
Winning Trades:   {n_wins:,} ({wr:.0%})
Losing Trades:    {n_losses:,} ({(1-wr):.0%})

Net PnL:          Rs {net_rs:+,.0f}
Avg Trade:        Rs {avg_rs:+,.0f}
Avg Win:          Rs {aw:+,.0f}
Avg Loss:         Rs {al:+,.0f}
W/L Ratio:        {wl:.1f}x
Profit Factor:    {pf_:.1f}x

Max Drawdown:     Rs {mdd_rs:,.0f} ({max_dd_pct:.1f}%)
Max Run-up:       {max_runup:.1f}%
Calmar Ratio:     {calmar:.1f}x
Sharpe Ratio:     {sharpe:.2f}

Max Consec Wins:  {max_consec_win}
Max Consec Losses: {max_consec_loss}

Avg Premium Paid: Rs {avg_prem*LOT:,.0f} ({avg_prem:.1f} pts)
Total Premium:    Rs {total_cost:,.0f}

Final Capital:    Rs {tb['MTM'].iloc[-1]:,.0f}
Total Return:     {(tb['CumRs'].iloc[-1]/INIT_CAP*100):+.1f}%
"""

print(stats_text)

# ========== PDF REPORT ==========
pdf=PdfPages("Option_1L_Backtest_Report.pdf")

# ===== PAGE 1: TITLE =====
fig=plt.figure(figsize=(8.5,11)); fig.patch.set_facecolor('#1a1a2e')
ax=fig.add_axes([0,0,1,1]); ax.axis('off')
ax.text(0.5,0.88,"Option Strategy Backtest Report",fontsize=22,fontweight='bold',color='white',ha='center',transform=ax.transAxes)
ax.text(0.5,0.82,f"Rs {INIT_CAP:,.0f} Capital | 1 Lot NIFTY ({LOT} units)",fontsize=13,color='#87ceeb',ha='center',transform=ax.transAxes)
ax.text(0.5,0.77,"Bullish Engulfing + TP30_Max7d + Prem<130",fontsize=11,color='#ccc',ha='center',transform=ax.transAxes)
ax.text(0.5,0.72,f"Period: 2021-2026 | {len(tb)} Trades",fontsize=10,color='#999',ha='center',transform=ax.transAxes)
ax.text(0.5,0.60,f"Net PnL: Rs {net_rs:+,.0f}",fontsize=16,color='#2ecc71',fontweight='bold',ha='center',transform=ax.transAxes)
ax.text(0.5,0.55,f"Win Rate: {wr:.0%} | W/L: {wl:.1f}x | Calmar: {calmar:.1f}x",fontsize=11,color='#aaa',ha='center',transform=ax.transAxes)
ax.text(0.5,0.50,f"Max DD: Rs {mdd_rs:,.0f} ({max_dd_pct:.1f}%) | Sharpe: {sharpe:.2f}",fontsize=11,color='#aaa',ha='center',transform=ax.transAxes)
ax.text(0.5,0.45,f"Final Capital: Rs {tb['MTM'].iloc[-1]:,.0f} ({(tb['CumRs'].iloc[-1]/INIT_CAP*100):+.1f}%)",fontsize=11,color='#f1c40f',ha='center',transform=ax.transAxes)
ax.text(0.5,0.35,"Strategy Rules:",fontsize=11,fontweight='bold',color='white',ha='center',transform=ax.transAxes)
rules_str=[
    "Buy 1 ATM CALL Weekly (atm_dist=0) on bullish engulfing signal",
    "Take Profit: Exit at +30 pts on option premium",
    "Time Stop: Exit at market after 7 trading days if TP not hit",
    "Entry Filter: Skip if CALL premium > 130 pts",
    "No stop loss (7-day time stop is the only loss control)"
]
for ri,rule in enumerate(rules_str):
    ax.text(0.5,0.30-ri*0.025,rule,fontsize=8,color='#ddd',ha='center',fontfamily='monospace',transform=ax.transAxes)
ax.text(0.5,0.08,"Generated: June 2026 | NIFTY50 Options Data",fontsize=9,color='#555',ha='center',transform=ax.transAxes)
pdf.savefig(fig); plt.close()

# ===== PAGE 2: STATISTICS SUMMARY =====
fig,ax=plt.subplots(figsize=(8.5,11)); ax.axis('off')
ax.set_title("Performance Summary",fontsize=14,fontweight='bold',pad=10)
stat_lines=stats_text.strip().split('\n')
for li,line in enumerate(stat_lines):
    y=0.95-li*0.024
    c='#333' if not line.startswith('=') else '#2c3e50'
    fw='bold' if line.startswith('=') or li==1 else 'normal'
    fs=9 if li<3 else 8
    ax.text(0.05,y,line,fontsize=fs,fontweight=fw,color=c,fontfamily='monospace',transform=ax.transAxes)
pdf.savefig(fig); plt.close()

# ===== PAGE 3: EQUITY CURVE + DD =====
fig,axes=plt.subplots(2,1,figsize=(8.5,8),gridspec_kw={'height_ratios':[3,1]})
fig.suptitle(f"Equity Curve & Drawdown (Rs {INIT_CAP:,.0f} start)",fontsize=12,fontweight='bold')

ax=axes[0]; eq=tb["MTM"].values
ax.plot(range(len(eq)),eq,color='#2ecc71',linewidth=1.2)
ax.fill_between(range(len(eq)),INIT_CAP,eq,alpha=0.15,color='#2ecc71')
ax.axhline(INIT_CAP,color='#333',linewidth=0.8,linestyle='--')
ax.set_ylabel("Portfolio Value (Rs)"); ax.grid(True,alpha=0.2)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'Rs{x:,.0f}'))
# Annotations
for yr in sorted(tb["Yr"].unique()):
    yr_trades=tb[tb["Yr"]==yr]
    if len(yr_trades)>0:
        idx=yr_trades.index[-1]; val=eq[idx]
        ax.annotate(f"{yr}\nRs{val:,.0f}",(idx,val),fontsize=6,ha='center',
                   xytext=(0,15),textcoords='offset points',arrowprops=dict(arrowstyle='->',color='#666',lw=0.5))

ax=axes[1]; dd_series=dd_rs
ax.fill_between(range(len(dd_series)),0,dd_series,alpha=0.5,color='#e74c3c')
ax.set_ylabel("Drawdown (Rs)"); ax.set_xlabel("Trade #")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'Rs{x:,.0f}'))
ax.grid(True,alpha=0.2)
plt.tight_layout()
pdf.savefig(fig); plt.close()

# ===== PAGE 4: YEARLY + MONTHLY =====
fig,axes=plt.subplots(2,2,figsize=(8.5,8))
fig.suptitle("Yearly & Monthly PnL (Rupees)",fontsize=12,fontweight='bold')

# Yearly bars
ax=axes[0,0]; yr=tb.groupby("Yr")["Rs"].agg(["sum","count",lambda x:(x>0).mean()])
yr.columns=["Net","Trades","WR"]
bars=ax.bar([str(int(y)) for y in yr.index],yr["Net"],color=['#2ecc71' if n>0 else '#e74c3c' for n in yr["Net"]])
ax.set_ylabel("Net PnL (Rs)"); ax.grid(True,alpha=0.2)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'Rs{x:,.0f}'))
for bar_obj,val in zip(bars,yr["Net"]):
    off=500 if val>0 else -800
    ax.text(bar_obj.get_x()+bar_obj.get_width()/2,bar_obj.get_height()+off,f"{val:+,.0f}",ha='center',fontsize=7,fontweight='bold')
for i,(yval,row) in enumerate(yr.iterrows()): ax.text(i,yr["Net"].min()-abs(yr["Net"].max())*0.12,f"WR:{row['WR']:.0%}",ha='center',fontsize=6)

# Yearly cumulative
ax=axes[0,1]; yr_cum=tb.groupby("Yr")["Rs"].sum().cumsum()
ax.plot([str(int(y)) for y in yr_cum.index],yr_cum.values,'bo-',markersize=5,linewidth=1.5)
ax.axhline(0,color='#333',linewidth=0.5)
ax.set_ylabel("Cumulative PnL"); ax.grid(True,alpha=0.2)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'Rs{x:,.0f}'))
for i,(yv) in enumerate(yr_cum.values):
    ax.text(i,yv+300,f"{yv:+,.0f}",ha='center',fontsize=7,fontweight='bold')
ax.set_title("Cumulative Yearly PnL")

# Monthly heatmap
ax=axes[1,0]; ax.set_title("Monthly PnL Heatmap (Rs)")
hm=tb.groupby(["Yr","Mo"])["Rs"].sum().unstack(fill_value=0)
years=sorted(hm.index)
for yi,yval in enumerate(years):
    for mi in range(1,13):
        val=hm.loc[yval,mi] if mi in hm.columns else 0
        c='#2ecc71' if val>0 else '#e74c3c' if val<0 else '#eee'
        ax.fill_between([mi-1,mi],[yi,yi],[yi+1,yi+1],color=c,alpha=1 if val!=0 else 0.3)
        if val!=0:
            txt=f"{val:+,.0f}"
            ax.text(mi-0.5,yi+0.5,txt,ha='center',va='center',fontsize=4.5,
                   color='white' if abs(val)>abs(hm.values).max()*0.3 else '#333')
ax.set_xlim(0,12); ax.set_ylim(len(years),0); ax.set_xticks(range(12)); ax.set_xticklabels(MONTHS,fontsize=6)
ax.set_yticks(range(len(years))); ax.set_yticklabels(years,fontsize=6)

# Monthly bars
ax=axes[1,1]; mo=tb.groupby("Mo")["Rs"].agg(["sum","count",lambda x:(x>0).mean()])
mo.columns=["Net","Trades","WR"]; mo=mo.reindex(range(1,13),fill_value=0)
bars=ax.bar(range(1,13),mo["Net"],color=['#2ecc71' if n>0 else '#e74c3c' for n in mo["Net"]])
ax.set_title("Monthly PnL"); ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTHS,fontsize=7)
ax.grid(True,alpha=0.2); ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'Rs{x:,.0f}'))
for i,val in enumerate(mo["Net"]):
    if val!=0: ax.text(i+1,val+(200 if val>0 else -400),f"{val:+,.0f}",ha='center',fontsize=6,fontweight='bold')

plt.tight_layout()
pdf.savefig(fig); plt.close()

# ===== PAGE 5: WIN/LOSS ANALYSIS =====
fig,axes=plt.subplots(1,3,figsize=(8.5,4))
fig.suptitle("Trade Distribution Analysis",fontsize=12,fontweight='bold')

ax=axes[0]; ax.hist(rs_arr,bins=20,color='#3498db',alpha=0.7,edgecolor='white')
ax.axvline(0,color='red',linewidth=1); ax.set_xlabel("PnL per Trade (Rs)"); ax.set_ylabel("Frequency")
ax.set_title(f"PnL Distribution (Avg: Rs{avg_rs:+,.0f})")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'{x:.0f}'))

ax=axes[1]; labels=['Wins','Losses']; sizes=[n_wins,n_losses]; colors=['#2ecc71','#e74c3c']
ax.pie(sizes,labels=labels,autopct='%1.0f%%',colors=colors,startangle=90,textprops={'fontsize':8})
ax.set_title(f"Win Rate: {wr:.0%}")

ax=axes[2]; streaks_df=pd.DataFrame({"Streak":win_streaks+[-s for s in loss_streaks], "Type":["Win"]*len(win_streaks)+["Loss"]*len(loss_streaks)})
streaks_vals=streaks_df["Streak"].value_counts().head(15).sort_index()
ax.barh([str(abs(s)) for s in streaks_vals.index],streaks_vals.values,color='#9b59b6',alpha=0.7)
ax.set_xlabel("Occurrences"); ax.set_title(f"Streak Lengths (Max: {max_consec_win}W/{max_consec_loss}L)")
plt.tight_layout()
pdf.savefig(fig); plt.close()

# ===== PAGE 6: WEEKDAY ANALYSIS =====
fig,ax=plt.subplots(figsize=(8.5,4))
fig.suptitle("Entry Day Analysis",fontsize=12,fontweight='bold')
wd=tb.groupby("Weekday")["Rs"].agg(["sum","count",lambda x:(x>0).mean()])
wd.columns=["Net","Trades","WR"]; wd=wd.reindex(range(5),fill_value=0)
wd_names=["Mon","Tue","Wed","Thu","Fri"]
bars=ax.bar(range(5),wd["Net"],color=['#2ecc71' if n>0 else '#e74c3c' for n in wd["Net"]])
ax.set_xticks(range(5)); ax.set_xticklabels(wd_names,fontsize=9)
ax.set_ylabel("Net PnL (Rs)"); ax.grid(True,alpha=0.2)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'Rs{x:,.0f}'))
for i,(idx,r) in enumerate(wd.iterrows()):
    if r["Trades"]>0:
        ax.text(i,r["Net"]+(100 if r["Net"]>0 else -200),f"Rs{r['Net']:+,.0f}\n{r['Trades']}T\nWR:{r['WR']:.0%}",ha='center',fontsize=7,fontweight='bold')
plt.tight_layout()
pdf.savefig(fig); plt.close()

# ===== PAGE 7: TRADE BOOK (first 50) =====
print("Trade book...")
fig,ax=plt.subplots(figsize=(8.5,11)); ax.axis('off')
ax.set_title(f"Trade Book (first 50 of {len(tb)} trades)",fontsize=12,fontweight='bold',pad=10)

tbl_cols=["#","Entry","Exit","EP","XP","Pts","Rs","CumRs","Return%"]
tbl_col_w=[0.03,0.14,0.14,0.07,0.07,0.06,0.10,0.11,0.10]
tx=np.cumsum([0.02]+tbl_col_w)

# Header
for ci,(col,tw) in enumerate(zip(tbl_cols,tbl_col_w)):
    ax.text(tx[ci],0.93,col,fontsize=7,fontweight='bold',fontfamily='monospace',transform=ax.transAxes)

show_n=min(50,len(tb))
for ti in range(show_n):
    r=tb.iloc[ti]; y2=0.91-ti*0.017
    vals=[str(r["#"]),str(r["Entry"]).split(".")[0][:19],str(r["Exit"]).split(".")[0][:19],
          f"{r['EntryPx']:.1f}",f"{r['ExitPx']:.1f}",
          f"{r['Pts']:+.1f}",f"Rs{r['Rs']:+,.0f}",f"Rs{r['CumRs']:+,.0f}",
          f"{r['Return%']:+.1f}%"]
    for ci,(v,tw) in enumerate(zip(vals,tbl_col_w)):
        c='#070' if r["Rs"]>0 else '#a00'
        ax.text(tx[ci],y2,v,fontsize=5.5,color=c,fontfamily='monospace',transform=ax.transAxes)

# Summary at bottom
ax.text(0.02,0.05,f"Total Trades: {len(tb)} | Net: Rs{net_rs:+,.0f} | "
    f"Avg: Rs{avg_rs:+,.0f} | WR: {wr:.0%} | MaxDD: Rs{mdd_rs:,.0f} ({max_dd_pct:.1f}%) | "
    f"Final: Rs{tb['MTM'].iloc[-1]:,.0f}",fontsize=7,fontfamily='monospace',transform=ax.transAxes,
    fontweight='bold',color='#2c3e50')

pdf.savefig(fig); plt.close()

# ===== PAGE 8: FULL TRADE BOOK (if <= 200, else pages 8-9) =====
if len(tb)>50:
    remaining=len(tb)-50
    n_pages=max(1,int(np.ceil(remaining/60)))
    for pg in range(n_pages):
        fig,ax=plt.subplots(figsize=(8.5,11)); ax.axis('off')
        ax.set_title(f"Trade Book (cont'd, trades {51+pg*60}-{min(50+(pg+1)*60,len(tb))})",fontsize=11,fontweight='bold',pad=10)
        
        for ci,(col,tw) in enumerate(zip(tbl_cols,tbl_col_w)):
            ax.text(tx[ci],0.93,col,fontsize=7,fontweight='bold',fontfamily='monospace',transform=ax.transAxes)
        
        start=50+pg*60; end=min(start+60,len(tb))
        for ti in range(start,end):
            r=tb.iloc[ti]; y2=0.91-(ti-start)*0.017
            vals=[str(r["#"]),str(r["Entry"]).split(".")[0][:19],str(r["Exit"]).split(".")[0][:19],
                  f"{r['EntryPx']:.1f}",f"{r['ExitPx']:.1f}",
                  f"{r['Pts']:+.1f}",f"Rs{r['Rs']:+,.0f}",f"Rs{r['CumRs']:+,.0f}",
                  f"{r['Return%']:+.1f}%"]
            for ci,(v,tw) in enumerate(zip(vals,tbl_col_w)):
                c='#070' if r["Rs"]>0 else '#a00'
                ax.text(tx[ci],y2,v,fontsize=5.5,color=c,fontfamily='monospace',transform=ax.transAxes)
        
        ax.text(0.02,0.04,f"Trades {start+1}-{end} of {len(tb)} | "
            f"Net: Rs{net_rs:+,.0f} | Final: Rs{tb['MTM'].iloc[-1]:,.0f}",
            fontsize=7,fontfamily='monospace',transform=ax.transAxes,fontweight='bold',color='#2c3e50')
        
        pdf.savefig(fig); plt.close()

pdf.close()
print(f"\nPDF saved: Option_1L_Backtest_Report.pdf ({len(tb)} trades, {8+n_pages if len(tb)>50 else 8} pages)")
print("Done!")
