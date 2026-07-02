"""
Single best strategy (Engulf_Raw CH55) - monthly heatmap + yearly report
"""
import pandas as pd, numpy as np, os, sys, io, warnings
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
plt.rcParams.update({"font.family":"Segoe UI, Arial, sans-serif","font.size":9,
                     "axes.edgecolor":"#333","axes.grid":True,"grid.alpha":0.3})

# Quick backtest: Engulf_Raw CH55 only
def compute_atr(m5):
    tr=pd.concat([m5["high"]-m5["low"],abs(m5["high"]-m5["close"].shift(1)),
                  abs(m5["low"]-m5["close"].shift(1))],axis=1).max(axis=1)
    return tr.ewm(span=14,min_periods=14,adjust=False).mean()

print("Loading...")
DATA={}
for sym in ["NIFTY50","SENSEX"]:
    h1=pd.read_csv(f"{sym}_ONE_HOUR.csv",parse_dates=["datetime"])
    m5=pd.read_csv(f"{sym}_FIVE_MINUTE.csv",parse_dates=["datetime"])
    for df in [h1,m5]:
        df["datetime"]=pd.to_datetime(df["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
        df.sort_values("datetime",inplace=True);df.reset_index(drop=True,inplace=True)
    atr5=compute_atr(m5)
    DATA[sym]={"h1":h1,"m5":m5,"m5_epoch":m5["datetime"].astype("int64").values,
               "m5_cl":m5["close"].values,"m5_lo":m5["low"].values,"m5_hi":m5["high"].values,
               "m5_atr":atr5.values,"tc":pd.Series(m5["datetime"]).dt.time.values}

CUT=pd.Timestamp("14:15").time()
def run():
    rows=[]
    for sym in ["NIFTY50","SENSEX"]:
        h1=DATA[sym]["h1"]; d=DATA[sym]
        b=(h1["close"]-h1["open"]).abs(); g=h1["close"]>h1["open"]; r=h1["close"]<h1["open"]
        me=d["m5_epoch"]; mc=d["m5_cl"]; ml=d["m5_lo"]; mh=d["m5_hi"]; ma=d["m5_atr"]; tc=d["tc"]
        for i in range(1,len(h1)):
            if not (r.iloc[i-1] and g.iloc[i]): continue
            if h1["open"].iloc[i]>h1["close"].iloc[i-1] or h1["close"].iloc[i]<h1["open"].iloc[i-1]: continue
            if b.iloc[i]<b.iloc[i-1]*0.5 or h1["datetime"].iloc[i].hour==9: continue
            lv=h1["high"].iloc[i]; ts=h1["datetime"].iloc[i]
            t_ep=ts.asm8.view("int64")
            idx=np.searchsorted(me,t_ep,side="right")
            if idx>=len(mc): continue
            bi=idx
            while bi<len(mc) and mc[bi]<=lv: bi+=1
            if bi>=len(mc)-1: continue
            ri=bi+1
            while ri<len(mc):
                if ml[ri]<lv and mc[ri]>lv and tc[ri]<CUT: break
                ri+=1
            if ri>=len(mc): continue
            ep=mc[ri]
            if ep-ml[ri]<=0: continue
            he=ep; exited=False
            for j in range(ri,len(mc)):
                ca=ma[j]
                if pd.isna(ca): continue
                if mh[j]>he: he=mh[j]
                if mc[j]<he-55*ca:
                    pnl=round(mc[j]-ep,1)
                    rows.append({"yr":ts.year,"mo":ts.month,"pnl":pnl,"sym":sym})
                    exited=True; break
            if not exited:
                rows.append({"yr":ts.year,"mo":ts.month,"pnl":round(mc[-1]-ep,1),"sym":sym})
    df=pd.DataFrame(rows)
    print(f"Trades: {len(df)}, Net: {df['pnl'].sum():+,.0f}")
    return df

df=run()
# Compute aggregates
yr_net=df.groupby("yr")["pnl"].sum()
yr_wr=df.groupby("yr")["pnl"].apply(lambda x:(x>0).sum()/len(x))
yr_n=df.groupby("yr").size()
mo_net=df.groupby("mo")["pnl"].sum()
mo_wr=df.groupby("mo")["pnl"].apply(lambda x:(x>0).sum()/len(x))
mo_n=df.groupby("mo").size()
yr_pivot=df.pivot_table(index="yr",columns="mo",values="pnl",aggfunc="sum").fillna(0)
yr_pivot.columns=[MON[c-1] for c in yr_pivot.columns]
yr_mo_n=df.pivot_table(index="yr",columns="mo",values="pnl",aggfunc="count").fillna(0).astype(int)

OUT=f"Best_Strategy_Viz_EngulfRaw_CH55.pdf"
with PdfPages(OUT) as pdf:
    # === 1: MONTHLY NET HEATMAP ===
    fig,ax=plt.subplots(figsize=(10,5))
    v=max(abs(mo_net.min()),abs(mo_net.max()))
    sns.heatmap(mo_net.values.reshape(1,12),ax=ax,cmap=sns.diverging_palette(10,130,s=85,l=45,as_cmap=True),
                center=0,vmin=-v,vmax=v,annot=True,fmt="+,.0f",linewidths=1,
                xticklabels=MON,yticklabels=["Net Pts"],
                cbar_kws={"shrink":0.5,"label":"Points"})
    ax.set_title("Engulf_Raw CH55: Monthly Net Return (2015-2026)",fontsize=13,fontweight="bold",pad=15)
    ax.tick_params(axis="x",rotation=45)
    fig.tight_layout(); pdf.savefig(fig,dpi=200); plt.close()

    # === 2: MONTHLY WR HEATMAP ===
    fig,ax=plt.subplots(figsize=(10,5))
    sns.heatmap(mo_wr.values.reshape(1,12),ax=ax,cmap="RdYlGn",vmin=0,vmax=1,
                annot=True,fmt=".0%",linewidths=1,
                xticklabels=MON,yticklabels=["Win Rate"],
                cbar_kws={"shrink":0.5,"label":"WR"})
    for i in range(12): ax.add_patch(plt.Rectangle((i,0),1,1,fill=False,edgecolor="gray",lw=1))
    ax.set_title("Engulf_Raw CH55: Monthly Win Rate (2015-2026)",fontsize=13,fontweight="bold",pad=15)
    ax.tick_params(axis="x",rotation=45)
    fig.tight_layout(); pdf.savefig(fig,dpi=200); plt.close()

    # === 3: TRADE COUNT HEATMAP ===
    fig,ax=plt.subplots(figsize=(10,5))
    sns.heatmap(mo_n.values.reshape(1,12),ax=ax,cmap="Blues",annot=True,fmt=".0f",linewidths=1,
                xticklabels=MON,yticklabels=["Trades"],
                cbar_kws={"shrink":0.5,"label":"Count"})
    ax.set_title("Engulf_Raw CH55: Monthly Trade Count (2015-2026)",fontsize=13,fontweight="bold",pad=15)
    ax.tick_params(axis="x",rotation=45)
    fig.tight_layout(); pdf.savefig(fig,dpi=200); plt.close()

    # === 4: MONTHLY BAR (NET + WR overlay) ===
    fig,ax1=plt.subplots(figsize=(12,5.5))
    bars=ax1.bar(MON,mo_net.values,color=["#2ecc71" if v>=0 else "#e74c3c" for v in mo_net.values],
                 alpha=0.85,edgecolor="gray",linewidth=0.5)
    for bar,v in zip(bars,mo_net.values):
        ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+(bar.get_height()*0.02 if v>=0 else bar.get_height()*0.02-8000),
                 f"{v:+,.0f}",ha="center",fontsize=8,fontweight="bold")
    ax1.axhline(y=0,color="black",linewidth=0.5)
    ax1.set_ylabel("Net Points",fontsize=10,color="#2c3e50")
    ax1.set_ylim(mo_net.min()*1.3,mo_net.max()*1.3)
    ax2=ax1.twinx()
    ax2.plot(MON,mo_wr.values,"o-",color="#3498db",linewidth=2,markersize=7,zorder=5)
    ax2.set_ylabel("Win Rate",fontsize=10,color="#3498db")
    ax2.set_ylim(0,1); ax2.axhline(y=0.5,color="#3498db",linewidth=0.5,linestyle="--",alpha=0.5)
    ax2.tick_params(axis="y",colors="#3498db")
    ax1.set_title("Engulf_Raw CH55: Monthly Net Return + Win Rate (2015-2026)",fontsize=13,fontweight="bold",pad=10)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax1.legend([Patch(color="#2ecc71",alpha=0.85),Line2D([],[],color="#3498db",marker="o",linewidth=2)],
               ["Net Pts","Win Rate"],loc="upper left",fontsize=9)
    fig.tight_layout(); pdf.savefig(fig,dpi=200); plt.close()

    # === 5: YEARLY BAR ===
    fig,ax=plt.subplots(figsize=(14,6))
    years=sorted(df["yr"].unique())
    yr_vals=[yr_net.get(y,0) for y in years]
    colors=["#2ecc71" if v>=0 else "#e74c3c" for v in yr_vals]
    bars=ax.bar([str(y) for y in years],yr_vals,color=colors,alpha=0.85,edgecolor="gray",linewidth=0.5)
    for bar,v in zip(bars,yr_vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+(bar.get_height()*0.02 if v>=0 else bar.get_height()*0.02-15000),
                f"{v:+,.0f}",ha="center",fontsize=9,fontweight="bold")
    ax.axhline(y=0,color="black",linewidth=0.5)
    total=sum(yr_vals)
    ax.text(0.5,0.95,f"TOTAL: {total:+,.0f} pts  |  Avg/yr: {total/len(years):+,.0f} pts",
            transform=ax.transAxes,ha="center",fontsize=11,
            bbox=dict(boxstyle="round",facecolor="lightyellow",edgecolor="gray"))
    ax.set_title("Engulf_Raw CH55: Yearly Net Return (2015-2026)",fontsize=13,fontweight="bold",pad=10)
    ax.set_ylabel("Net Points"); ax.set_xlabel("Year")
    ax.grid(axis="y",alpha=0.3)
    fig.tight_layout(); pdf.savefig(fig,dpi=200); plt.close()

    # === 6: CUMULATIVE EQUITY CURVE ===
    fig,ax=plt.subplots(figsize=(14,6))
    cum=df.sort_values(["sym","yr","mo"])["pnl"].cumsum()
    ax.plot(range(len(cum)),cum,color="#2c3e50",linewidth=1.5)
    ax.fill_between(range(len(cum)),0,cum,alpha=0.15,color="#2ecc71" if cum.iloc[-1]>=0 else "#e74c3c")
    ax.axhline(y=0,color="gray",linewidth=0.5,linestyle="--")
    total=cum.iloc[-1]
    ax.text(0.02,0.95,f"Final Equity: {total:+,.0f} pts  |  Trades: {len(df)}",
            transform=ax.transAxes,fontsize=11,va="top",
            bbox=dict(boxstyle="round",facecolor="lightyellow",edgecolor="gray"))
    ax.set_title("Engulf_Raw CH55: Cumulative Equity Curve (2015-2026)",fontsize=13,fontweight="bold",pad=10)
    ax.set_ylabel("Cumulative Points"); ax.set_xlabel("Trade Sequence (chronological)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); pdf.savefig(fig,dpi=200); plt.close()

    # === 7: YEAR × MONTH MATRIX (full heatmap) ===
    fig,ax=plt.subplots(figsize=(12,7))
    yp=yr_pivot.copy()
    v2=max(abs(yp.values.min()),abs(yp.values.max()))
    sns.heatmap(yp,ax=ax,cmap=sns.diverging_palette(10,130,s=85,l=45,as_cmap=True),
                center=0,vmin=-v2,vmax=v2,annot=True,fmt="+,.0f",linewidths=0.5,
                cbar_kws={"shrink":0.6,"label":"Points"})
    ax.set_title("Engulf_Raw CH55: Year × Month Net Return",fontsize=13,fontweight="bold",pad=15)
    ax.set_ylabel("Year"); ax.set_xlabel("Month")
    ax.tick_params(axis="x",rotation=45); ax.tick_params(axis="y",rotation=0)
    fig.tight_layout(); pdf.savefig(fig,dpi=200); plt.close()

    # === 8: MONTHLY TREND (all years overlaid) ===
    fig,ax=plt.subplots(figsize=(12,6))
    for yr in years:
        row=[yr_pivot.loc[yr,c] if c in yr_pivot.columns else 0 for c in MON]
        ax.plot(MON,row,"o-",label=str(yr),alpha=0.6,linewidth=1,markersize=4)
    ax.plot(MON,mo_net.values,"o-",color="black",linewidth=3,markersize=8,label="AVERAGE",zorder=10)
    ax.axhline(y=0,color="gray",linewidth=0.5,linestyle="--")
    ax.set_title("Engulf_Raw CH55: Monthly Net by Year (2015-2026)",fontsize=13,fontweight="bold",pad=10)
    ax.set_ylabel("Net Points"); ax.set_xlabel("Month")
    ax.legend(ncol=3,fontsize=8,loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout(); pdf.savefig(fig,dpi=200); plt.close()

    # === 9: YEARLY WR BAR ===
    fig,ax=plt.subplots(figsize=(14,5))
    wr_vals=[yr_wr.get(y,0) for y in years]
    colors_wr=["#2ecc71" if v>=0.5 else "#e74c3c" for v in wr_vals]
    bars=ax.bar([str(y) for y in years],wr_vals,color=colors_wr,alpha=0.85,edgecolor="gray",linewidth=0.5)
    for bar,v in zip(bars,wr_vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,f"{v:.0%}",
                ha="center",fontsize=9,fontweight="bold")
    ax.axhline(y=0.5,color="gray",linewidth=0.5,linestyle="--")
    ax.set_ylim(0,1); ax.set_title("Engulf_Raw CH55: Yearly Win Rate",fontsize=13,fontweight="bold",pad=10)
    ax.set_ylabel("Win Rate"); ax.set_xlabel("Year"); ax.grid(axis="y",alpha=0.3)
    fig.tight_layout(); pdf.savefig(fig,dpi=200); plt.close()

    # === 10: YEARLY TRADE COUNT ===
    fig,ax=plt.subplots(figsize=(14,5))
    n_vals=[yr_n.get(y,0) for y in years]
    ax.bar([str(y) for y in years],n_vals,color="#3498db",alpha=0.85,edgecolor="gray",linewidth=0.5)
    for xv,nv in zip(range(len(years)),n_vals):
        ax.text(xv,nv+1,str(nv),ha="center",fontsize=9,fontweight="bold")
    ax.set_title("Engulf_Raw CH55: Yearly Trade Count",fontsize=13,fontweight="bold",pad=10)
    ax.set_ylabel("Trades"); ax.set_xlabel("Year"); ax.grid(axis="y",alpha=0.3)
    fig.tight_layout(); pdf.savefig(fig,dpi=200); plt.close()

    # === 11: INSTRUMENT BREAKDOWN ===
    fig,ax=plt.subplots(figsize=(10,5))
    sym_n=df.groupby("sym").size(); sym_net=df.groupby("sym")["pnl"].sum()
    x=np.arange(2); w=0.35
    ax.bar(x-w/2,sym_n.values,w,color=["#3498db","#e74c3c"],alpha=0.85,label="Trades")
    ax2_=ax.twinx()
    ax2_.bar(x+w/2,sym_net.values,w,color=["#2ecc71","#f39c12"],alpha=0.85,label="Net Pts")
    ax.set_xticks(x); ax.set_xticklabels(sym_n.index)
    ax.set_ylabel("Trades",color="#3498db"); ax2_.set_ylabel("Net Points",color="#2ecc71")
    ax.set_title("Engulf_Raw CH55: NIFTY50 vs SENSEX",fontsize=13,fontweight="bold",pad=10)
    lines1=[plt.Rectangle((0,0),1,1,color="#3498db",alpha=0.85),
            plt.Rectangle((0,0),1,1,color="#2ecc71",alpha=0.85)]
    ax.legend(lines1,["Trades","Net Pts"],loc="upper left",fontsize=9)
    fig.tight_layout(); pdf.savefig(fig,dpi=200); plt.close()

op=os.path.join(os.path.dirname(__file__)or".",OUT)
print(f"\nSaved: {op}")
print(f"Size: {os.path.getsize(op):,} bytes")
print(f"Pages: 11 (all Engulf_Raw CH55)")
