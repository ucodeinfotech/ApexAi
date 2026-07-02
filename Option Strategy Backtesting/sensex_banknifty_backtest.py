"""
SENSEX & BANKNIFTY spot backtest: bullish engulfing + CH55
Full PDF with trade book, Rs 1L capital, 1 lot
"""
import pandas as pd, numpy as np, os, sys, io, warnings, itertools
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\pc\Downloads\stock hist data\Option Strategy Backtesting")
MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

plt.rcParams.update({'font.size':8,'axes.titlesize':11,'axes.labelsize':9,
    'figure.dpi':120,'savefig.dpi':150,'font.family':'sans-serif'})

LOTS = {"SENSEX": 50, "BANKNIFTY": 25}

def run_backtest(name, file_h1, file_m5, lot_size):
    """Run bullish engulfing + CH55 backtest, return trade book"""
    print(f"\n=== {name} ===")
    h1=pd.read_csv(file_h1,parse_dates=["datetime"])
    m5=pd.read_csv(file_m5,parse_dates=["datetime"])
    for d in [h1,m5]:
        d["datetime"]=pd.to_datetime(d["datetime"],utc=True).dt.tz_convert("Asia/Kolkata")
        d.sort_values("datetime",inplace=True);d.reset_index(drop=True,inplace=True)
    
    # ATR on 5-min
    tr=pd.concat([m5["high"]-m5["low"],abs(m5["high"]-m5["close"].shift(1)),abs(m5["low"]-m5["close"].shift(1))],axis=1).max(axis=1)
    atr5=tr.ewm(span=14,min_periods=14,adjust=False).mean()
    me=m5["datetime"].astype("int64").values
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
        ed=m5["datetime"].iloc[ri]; ep=m5["close"].iloc[ri]
        if ep-m5["low"].iloc[ri]<=0: continue
        he=ep
        for j in range(ri,len(m5["close"])):
            ca=atr5.iloc[j]
            if pd.isna(ca): continue
            if m5["high"].iloc[j]>he: he=m5["high"].iloc[j]
            if m5["close"].iloc[j]<he-55*ca:
                trades.append({"entry_dt":ed,"exit_dt":m5["datetime"].iloc[j],"entry":ep,
                    "exit":m5["close"].iloc[j],"yr":int(ts.year),"mo":int(ts.month),
                    "weekday":int(ed.weekday())}); break
    
    tb=pd.DataFrame(trades)
    tb["Pts"]=round(tb["exit"]-tb["entry"],1)
    tb["Rs"]=round(tb["Pts"]*lot_size,2)
    tb["CumRs"]=tb["Rs"].cumsum()
    INIT=100000.0
    tb["MTM"]=INIT+tb["CumRs"]
    tb["Return%"]=round(tb["CumRs"]/INIT*100,2)
    # Add trade #
    tb.insert(0,"#",range(1,len(tb)+1))
    
    print(f"Trades: {len(tb)}, Net: Rs {tb['CumRs'].iloc[-1]:+,.0f}, WR: {(tb['Rs']>0).mean():.0%}")
    return tb

def generate_pdf(tb, name, lot, INIT=100000.0):
    """Generate comprehensive PDF"""
    rs=tb["Rs"].values; net=rs.sum(); wr=(rs>0).mean(); n=len(tb)
    aw=rs[rs>0].mean() if (rs>0).sum()>0 else 0
    al=rs[rs<0].mean() if (rs<0).sum()>0 else 0
    wl=aw/abs(al) if al!=0 else 999; pf=rs[rs>0].sum()/abs(rs[rs<0].sum()) if (rs<0).sum()>0 else 999
    cum=np.cumsum(rs); mx=np.maximum.accumulate(cum); dd=mx-cum; mdd=dd.max()
    calmar=net/mdd if mdd>0 else 999; sharpe=rs.mean()/rs.std()*np.sqrt(252) if rs.std()>0 else 0
    max_dd_pct=(dd/(INIT+mx)).max()*100
    avg_rs=rs.mean(); n_wins=(rs>0).sum(); n_losses=(rs<0).sum()
    win_sk=[sum(1 for _ in g) for k,g in itertools.groupby(rs>0) if k]
    loss_sk=[sum(1 for _ in g) for k,g in itertools.groupby(rs<=0) if k]
    mcw=max(win_sk) if win_sk else 0; mcl=max(loss_sk) if loss_sk else 0
    
    pdf=PdfPages(f"{name}_1L_Backtest_Report.pdf")
    plt.rcParams.update({'font.size':8,'axes.titlesize':11,'axes.labelsize':9})
    
    # PAGE 1: TITLE
    fig=plt.figure(figsize=(8.5,11)); fig.patch.set_facecolor('#1a1a2e')
    ax=fig.add_axes([0,0,1,1]); ax.axis('off')
    ax.text(0.5,0.88,f"{name} Backtest Report",fontsize=22,fontweight='bold',color='white',ha='center',transform=ax.transAxes)
    ax.text(0.5,0.82,f"Rs {INIT:,.0f} Capital | 1 Lot ({lot} units)",fontsize=13,color='#87ceeb',ha='center',transform=ax.transAxes)
    ax.text(0.5,0.77,"Bullish Engulfing + CH55 Trail Exit",fontsize=11,color='#ccc',ha='center',transform=ax.transAxes)
    ax.text(0.5,0.72,f"Period: {tb['entry_dt'].min().year}-{tb['entry_dt'].max().year} | {n} Trades",fontsize=10,color='#999',ha='center',transform=ax.transAxes)
    ax.text(0.5,0.60,f"Net PnL: Rs {net:+,.0f}",fontsize=16,color='#2ecc71',fontweight='bold',ha='center',transform=ax.transAxes)
    ax.text(0.5,0.55,f"Win Rate: {wr:.0%} | W/L: {wl:.1f}x | Calmar: {calmar:.1f}x",fontsize=11,color='#aaa',ha='center',transform=ax.transAxes)
    ax.text(0.5,0.50,f"Max DD: Rs {mdd:,.0f} ({max_dd_pct:.1f}%) | Sharpe: {sharpe:.2f}",fontsize=11,color='#aaa',ha='center',transform=ax.transAxes)
    ax.text(0.5,0.45,f"Final Capital: Rs {tb['MTM'].iloc[-1]:,.0f} ({(tb['CumRs'].iloc[-1]/INIT*100):+.1f}%)",fontsize=11,color='#f1c40f',ha='center',transform=ax.transAxes)
    ax.text(0.5,0.08,"Generated: June 2026 | Spot 5-min Data",fontsize=9,color='#555',ha='center',transform=ax.transAxes)
    pdf.savefig(fig); plt.close()
    
    # PAGE 2: STATS
    fig,ax=plt.subplots(figsize=(8.5,11)); ax.axis('off')
    ax.set_title("Performance Summary",fontsize=14,fontweight='bold',pad=10)
    stats=f"""
========== {name} STRATEGY STATISTICS (Rupees) ==========
Instrument:       {name} (1 lot = {lot} units)
Capital:          Rs {INIT:,.0f}
Strategy:         Bullish Engulfing + CH55 Trail Exit

Total Trades:     {n:,}
Winning Trades:   {n_wins:,} ({wr:.0%})
Losing Trades:    {n_losses:,} ({(1-wr):.0%})

Net PnL:          Rs {net:+,.0f}
Avg Trade:        Rs {avg_rs:+,.0f}
Avg Win:          Rs {aw:+,.0f}
Avg Loss:         Rs {al:+,.0f}
W/L Ratio:        {wl:.1f}x
Profit Factor:    {pf:.1f}x

Max Drawdown:     Rs {mdd:,.0f} ({max_dd_pct:.1f}%)
Calmar Ratio:     {calmar:.1f}x
Sharpe Ratio:     {sharpe:.2f}

Max Consec Wins:  {mcw}
Max Consec Losses: {mcl}

Final Capital:    Rs {tb['MTM'].iloc[-1]:,.0f}
Total Return:     {(tb['CumRs'].iloc[-1]/INIT*100):+.1f}%
"""
    for li,line in enumerate(stats.strip().split('\n')):
        y=0.95-li*0.024; c='#333' if not line.startswith('=') else '#2c3e50'
        fw='bold' if line.startswith('=') or 'STATISTICS' in line else 'normal'
        fs=9 if li<3 else 8
        ax.text(0.05,y,line,fontsize=fs,fontweight=fw,color=c,fontfamily='monospace',transform=ax.transAxes)
    pdf.savefig(fig); plt.close()
    
    # PAGE 3: EQUITY + DD
    fig,axes=plt.subplots(2,1,figsize=(8.5,8),gridspec_kw={'height_ratios':[3,1]})
    fig.suptitle(f"Equity Curve & Drawdown (Rs {INIT:,.0f} start)",fontsize=12,fontweight='bold')
    ax=axes[0]; eq=tb["MTM"].values
    ax.plot(range(len(eq)),eq,color='#2ecc71',linewidth=1.2)
    ax.fill_between(range(len(eq)),INIT,eq,alpha=0.15,color='#2ecc71')
    ax.axhline(INIT,color='#333',linewidth=0.8,linestyle='--')
    ax.set_ylabel("Portfolio (Rs)"); ax.grid(True,alpha=0.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'Rs{x:,.0f}'))
    for yr_ in sorted(tb["yr"].unique()):
        sub=tb[tb["yr"]==yr_]
        if len(sub)>0:
            ax.annotate(f"{yr_}\nRs{sub['MTM'].iloc[-1]:,.0f}",(sub.index[-1],sub['MTM'].iloc[-1]),
                       fontsize=6,ha='center',xytext=(0,12),textcoords='offset points',arrowprops=dict(arrowstyle='->',color='#666',lw=0.5))
    ax=axes[1]
    ax.fill_between(range(len(dd)),0,dd,alpha=0.5,color='#e74c3c')
    ax.set_ylabel("DD (Rs)"); ax.set_xlabel("Trade #"); ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'Rs{x:,.0f}'))
    ax.grid(True,alpha=0.2)
    plt.tight_layout(); pdf.savefig(fig); plt.close()
    
    # PAGE 4: YEARLY + MONTHLY
    fig,axes=plt.subplots(2,2,figsize=(8.5,8))
    fig.suptitle("Yearly & Monthly PnL (Rupees)",fontsize=12,fontweight='bold')
    ax=axes[0,0]; yr=tb.groupby("yr")["Rs"].agg(["sum","count",lambda x:(x>0).mean()])
    yr.columns=["Net","Trades","WR"]
    bars=ax.bar([str(int(y)) for y in yr.index],yr["Net"],color=['#2ecc71' if n>0 else '#e74c3c' for n in yr["Net"]])
    ax.set_ylabel("Net PnL (Rs)"); ax.grid(True,alpha=0.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'Rs{x:,.0f}'))
    for b,val in zip(bars,yr["Net"]):
        off=net*0.02 if val>0 else -net*0.04; ax.text(b.get_x()+b.get_width()/2,b.get_height()+off,f"{val:+,.0f}",ha='center',fontsize=7,fontweight='bold')
    for i,(yval,row) in enumerate(yr.iterrows()): ax.text(i,yr["Net"].min()-abs(yr["Net"].max())*0.1,f"WR:{row['WR']:.0%}",ha='center',fontsize=6)
    ax=axes[0,1]; yr_cum=tb.groupby("yr")["Rs"].sum().cumsum()
    ax.plot([str(int(y)) for y in yr_cum.index],yr_cum.values,'bo-',markersize=5)
    ax.axhline(0,color='#333',linewidth=0.5); ax.set_ylabel("Cum PnL"); ax.grid(True,alpha=0.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'Rs{x:,.0f}'))
    for i,v in enumerate(yr_cum.values): ax.text(i,v+net*0.02,f"{v:+,.0f}",ha='center',fontsize=7,fontweight='bold')
    ax=axes[1,0]; ax.set_title("Monthly Heatmap (Rs)")
    hm=tb.groupby(["yr","mo"])["Rs"].sum().unstack(fill_value=0); years=sorted(hm.index)
    for yi,yval in enumerate(years):
        for mi in range(1,13):
            val=hm.loc[yval,mi] if mi in hm.columns else 0
            c='#2ecc71' if val>0 else '#e74c3c' if val<0 else '#eee'
            ax.fill_between([mi-1,mi],[yi,yi],[yi+1,yi+1],color=c,alpha=1 if val!=0 else 0.3)
            if val!=0: ax.text(mi-0.5,yi+0.5,f"{val:+,.0f}",ha='center',va='center',fontsize=4.5,
                    color='white' if abs(val)>abs(hm.values).max()*0.3 else '#333')
    ax.set_xlim(0,12); ax.set_ylim(len(years),0); ax.set_xticks(range(12)); ax.set_xticklabels(MONTHS,fontsize=6)
    ax.set_yticks(range(len(years))); ax.set_yticklabels(years,fontsize=6)
    ax=axes[1,1]; mo=tb.groupby("mo")["Rs"].agg(["sum","count",lambda x:(x>0).mean()])
    mo.columns=["Net","Trades","WR"]; mo=mo.reindex(range(1,13),fill_value=0)
    ax.bar(range(1,13),mo["Net"],color=['#2ecc71' if n>0 else '#e74c3c' for n in mo["Net"]])
    ax.set_title("Monthly PnL"); ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTHS,fontsize=7); ax.grid(True,alpha=0.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'Rs{x:,.0f}'))
    for i,val in enumerate(mo["Net"]):
        if val!=0: ax.text(i+1,val+(net*0.02 if val>0 else -net*0.04),f"{val:+,.0f}",ha='center',fontsize=6,fontweight='bold')
    plt.tight_layout(); pdf.savefig(fig); plt.close()
    
    # PAGE 5: DISTRIBUTION
    fig,axes=plt.subplots(1,3,figsize=(8.5,4))
    fig.suptitle("Trade Distribution",fontsize=12,fontweight='bold')
    ax=axes[0]; ax.hist(rs,bins=20,color='#3498db',alpha=0.7,edgecolor='white')
    ax.axvline(0,color='red',linewidth=1); ax.set_xlabel("PnL (Rs)"); ax.set_ylabel("Freq"); ax.set_title(f"Avg: Rs{avg_rs:+,.0f}")
    ax=axes[1]; ax.pie([n_wins,n_losses],labels=['Wins','Losses'],autopct='%1.0f%%',colors=['#2ecc71','#e74c3c'],startangle=90,textprops={'fontsize':8})
    ax.set_title(f"WR: {wr:.0%}")
    ax=axes[2]; sk_df=pd.DataFrame({"S":win_sk+[-s for s in loss_sk],"T":["W"]*len(win_sk)+["L"]*len(loss_sk)})
    sv=sk_df["S"].value_counts().head(15).sort_index()
    ax.barh([str(abs(s)) for s in sv.index],sv.values,color='#9b59b6',alpha=0.7)
    ax.set_xlabel("Occurrences"); ax.set_title(f"Streaks ({mcw}W/{mcl}L)")
    plt.tight_layout(); pdf.savefig(fig); plt.close()
    
    # PAGE 6: WEEKDAY
    fig,ax=plt.subplots(figsize=(8.5,4))
    fig.suptitle("Entry Day Analysis",fontsize=12,fontweight='bold')
    wd=tb.groupby("weekday")["Rs"].agg(["sum","count",lambda x:(x>0).mean()])
    wd.columns=["Net","Trades","WR"]; wd=wd.reindex(range(5),fill_value=0)
    wdn=["Mon","Tue","Wed","Thu","Fri"]
    bars=ax.bar(range(5),wd["Net"],color=['#2ecc71' if n>0 else '#e74c3c' for n in wd["Net"]])
    ax.set_xticks(range(5)); ax.set_xticklabels(wdn,fontsize=9); ax.set_ylabel("Net PnL"); ax.grid(True,alpha=0.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'Rs{x:,.0f}'))
    for i,(idx,r) in enumerate(wd.iterrows()):
        if r["Trades"]>0: ax.text(i,r["Net"]+(net*0.015 if r["Net"]>0 else -net*0.03),
                f"Rs{r['Net']:+,.0f}\n{r['Trades']}T\nWR:{r['WR']:.0%}",ha='center',fontsize=7,fontweight='bold')
    plt.tight_layout(); pdf.savefig(fig); plt.close()
    
    # TRADE BOOK PAGES
    tcols=["#","Entry","Exit","EP","XP","Pts","Rs","CumRs","Return%"]
    tcw=[0.03,0.14,0.14,0.07,0.07,0.06,0.10,0.11,0.10]
    tx=np.cumsum([0.02]+tcw)
    per_page=55
    
    npages=max(1,int(np.ceil(n/per_page)))
    for pg in range(npages):
        fig,ax=plt.subplots(figsize=(8.5,11)); ax.axis('off')
        s=pg*per_page; e=min(s+per_page,n)
        ax.set_title(f"Trade Book ({s+1}-{e} of {n})",fontsize=11,fontweight='bold',pad=10)
        for ci,(col,tw) in enumerate(zip(tcols,tcw)):
            ax.text(tx[ci],0.93,col,fontsize=7,fontweight='bold',fontfamily='monospace',transform=ax.transAxes)
        for ti in range(s,e):
            r=tb.iloc[ti]; y2=0.91-(ti-s)*0.017
            vals=[str(r["#"]),str(r["entry_dt"]).split("+")[0][:19],str(r["exit_dt"]).split("+")[0][:19],
                  f"{r['entry']:.1f}",f"{r['exit']:.1f}",f"{r['Pts']:+.1f}",
                  f"Rs{r['Rs']:+,.0f}",f"Rs{r['CumRs']:+,.0f}",f"{r['Return%']:+.1f}%"]
            for ci2,(v,tw) in enumerate(zip(vals,tcw)):
                ax.text(tx[ci2],y2,v,fontsize=5.5,color='#070' if r["Rs"]>0 else '#a00',fontfamily='monospace',transform=ax.transAxes)
        ax.text(0.02,0.04,f"Trades {s+1}-{e} | Net: Rs{net:+,.0f} | Final: Rs{tb['MTM'].iloc[-1]:,.0f}",
                fontsize=7,fontfamily='monospace',transform=ax.transAxes,fontweight='bold',color='#2c3e50')
        pdf.savefig(fig); plt.close()
    
    pdf.close()
    print(f"PDF saved: {name}_1L_Backtest_Report.pdf ({n} trades, {npages+6} pages)")

# === RUN ===
for name,f1,f5 in [("SENSEX","SENSEX_ONE_HOUR.csv","SENSEX_FIVE_MINUTE.csv"),
                    ("BANKNIFTY","BANKNIFTY_ONE_HOUR.csv","BANKNIFTY_FIVE_MINUTE.csv")]:
    tb=run_backtest(name,f1,f5,LOTS[name])
    generate_pdf(tb,name,LOTS[name])
    print(f"Done: {name}")

print("\nAll done!")
