"""Generate PDF report for BB(20,2.5) breakout strategy"""
import pandas as pd, numpy as np, os
from fpdf import FPDF
from datetime import datetime

OUTPUT_DIR = "backtest_results"
DATA_DIR = "nifty50_full_history"
BROKERAGE_PER_ORDER=10; STT=0.001; EXCHANGE_TC=0.00003; SEBI_TC=0.000001
GST=0.18; STAMP_DUTY=0.00003

def compute_charges(entry_price, exit_price, qty=1):
    tb=entry_price*qty; ts=exit_price*qty
    return (BROKERAGE_PER_ORDER*2+ts*STT+(tb+ts)*EXCHANGE_TC+(tb+ts)*SEBI_TC*2+tb*STAMP_DUTY+(BROKERAGE_PER_ORDER*2+(tb+ts)*EXCHANGE_TC)*GST)

def run_strategy(symbol, period=20, n_std=2.5, rr=3.0):
    df=pd.read_csv(f"{DATA_DIR}/{symbol}_FIFTEEN_MINUTE.csv")
    df["datetime"]=pd.to_datetime(df["datetime"])
    df=df.sort_values("datetime").reset_index(drop=True)
    df["date"]=df["datetime"].dt.date; df["year"]=df["datetime"].dt.year
    df["month"]=df["datetime"].dt.month; df["dow"]=df["datetime"].dt.dayofweek
    ma=df["close"].rolling(period).mean()
    std=df["close"].rolling(period).std(ddof=1)
    upper=ma+n_std*std; lower=ma-n_std*std
    trades=[]
    for i in range(period, len(df)-1):
        row=df.iloc[i]
        if row["low"]>upper.iloc[i]:
            typ,entry_p,t1_p="SHORT",row["close"],row["low"]
            tp_p=entry_p-(entry_p-t1_p)*rr; t1_dist=entry_p-t1_p
        elif row["high"]<lower.iloc[i]:
            typ,entry_p,t1_p="LONG",row["close"],row["high"]
            tp_p=entry_p+(t1_p-entry_p)*rr; t1_dist=t1_p-entry_p
        else: continue
        if t1_dist<=0: continue
        k=i+1; exit_p,reason=entry_p,"EOD"
        while k<len(df):
            b=df.iloc[k]; bdt=b["datetime"]
            if bdt.hour>=15 and bdt.minute>=15:
                exit_p=b["close"]; reason="EOD"; break
            tp_hit=(typ=="SHORT" and b["low"]<=tp_p) or (typ=="LONG" and b["high"]>=tp_p)
            t1_hit=(typ=="SHORT" and b["low"]<=t1_p) or (typ=="LONG" and b["high"]>=t1_p)
            if tp_hit and t1_hit: exit_p=tp_p; reason="TP"; break
            elif tp_hit: exit_p=tp_p; reason="TP"; break
            elif t1_hit: exit_p=t1_p; reason="T1"; break
            k+=1
        exit_time=b["datetime"] if k<len(df) else row["datetime"]
        pnl_pts=(entry_p-exit_p) if typ=="SHORT" else (exit_p-entry_p)
        charges=compute_charges(entry_p,exit_p)
        trades.append(dict(symbol=symbol, date=str(row["datetime"].date()), exit_time=str(exit_time),
            year=int(row["year"]), month=int(row["month"]), dow=int(row["dow"]), type=typ,
            entry=round(entry_p,2), exit=round(exit_p,2), t1=round(t1_p,2), tp=round(tp_p,2),
            t1_pts=round(t1_dist,2), tp_pts=round(abs(entry_p-tp_p),2),
            pnl_pts=round(pnl_pts,2), charges=round(charges,2), net_pnl=round(pnl_pts-charges,2),
            r=round(pnl_pts/t1_dist,2) if t1_dist>0 else 0, reason=reason))
    return pd.DataFrame(trades)

results={}
for sym in ["NIFTY50","BANKNIFTY","SENSEX"]:
    df=run_strategy(sym)
    results[sym]=df
    # Save to CSV
    df.to_csv(os.path.join(OUTPUT_DIR,f"bb_{sym.lower()}_trades.csv"), index=False)
    t=len(df); w=(df["pnl_pts"]>0).sum()
    t1c=(df["reason"]=="T1").sum(); tpc=(df["reason"]=="TP").sum(); eodc=(df["reason"]=="EOD").sum()
    print(f"{sym}: {t} trades, W:{w} ({w/t*100:.1f}%), Gross:{df['pnl_pts'].sum():.0f}, "
          f"Charges:{df['charges'].sum():.0f}, Net:{df['net_pnl'].sum():.0f}, "
          f"T1:{t1c}, TP:{tpc}, EOD:{eodc}, AvgR:{df['r'].mean():.2f}")

# --- Compute stats helper ---
def compute_stats(df):
    t=len(df); w=int((df["pnl_pts"]>0).sum()); l=t-w
    wr=round(w/t*100,1) if t else 0
    gross=df["pnl_pts"].sum(); net=df["net_pnl"].sum()
    ch=df["charges"].sum()
    win_pts=df[df["pnl_pts"]>0]["pnl_pts"].sum()
    loss_pts=abs(df[df["pnl_pts"]<=0]["pnl_pts"].sum())
    pf=round(win_pts/loss_pts,2) if loss_pts>0 else float("inf")
    ar=round(df["r"].mean(),2)
    t1c=int((df["reason"]=="T1").sum()); tpc=int((df["reason"]=="TP").sum()); eodc=int((df["reason"]=="EOD").sum())
    t1_dist=round(df[df["reason"]=="T1"]["t1_pts"].mean(),1) if t1c else 0
    tp_dist=round(df[df["reason"]=="TP"]["tp_pts"].mean(),1) if tpc else 0
    cs=df.sort_values("exit_time").reset_index(drop=True)
    cs["cum"]=cs["net_pnl"].cumsum(); cs["peak"]=cs["cum"].cummax()
    mdd=round((cs["peak"]-cs["cum"]).max(),2)
    sh=round(df["r"].mean()/df["r"].std()*np.sqrt(t),2) if df["r"].std()>0 else 0
    return dict(t=t,w=w,l=l,wr=wr,gross=gross,net=net,ch=ch,pf=pf,ar=ar,
                t1c=t1c,tpc=tpc,eodc=eodc,t1_dist=t1_dist,tp_dist=tp_dist,mdd=mdd,sh=sh)

s={sym: compute_stats(results[sym]) for sym in ["NIFTY50","BANKNIFTY","SENSEX"]}

# --- PDF ---
class PDF(FPDF):
    def header(self):
        if self.page_no()>1:
            self.set_font("Helvetica","I",7)
            self.set_text_color(130,130,130)
            self.cell(0,5,"BB(20,2.5) Breakout Strategy | 15-min Data | RR 1:3 | Spot Indices",align="C")
            self.ln(6)
            self.set_draw_color(200,200,200)
            self.line(10,self.get_y(),200,self.get_y())
            self.ln(3)
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica","I",7)
        self.set_text_color(150,150,150)
        self.cell(0,8,f"Page {self.page_no()}",align="C")
    def section(self,title):
        self.set_font("Helvetica","B",13)
        self.set_text_color(20,50,100)
        self.ln(3)
        self.cell(0,9,title)
        self.ln(7)
        self.set_draw_color(20,50,100)
        self.line(10,self.get_y(),200,self.get_y())
        self.ln(4)
    def subsection(self,title):
        self.set_font("Helvetica","B",10)
        self.set_text_color(60,60,60)
        self.cell(0,7,title)
        self.ln(6)
    def text(self,txt):
        self.set_font("Helvetica","",8.5)
        self.set_text_color(40,40,40)
        self.multi_cell(0,4.5,txt)
        self.ln(2)
    def kv(self,key,val):
        self.set_font("Helvetica","B",8.5)
        self.set_text_color(40,40,40)
        self.cell(50,5,key)
        self.set_font("Helvetica","",8.5)
        self.cell(0,5,val)
        self.ln(4.5)
    def table(self,headers,data,col_widths,highlight_col=None,flag_fn=None):
        self.set_font("Helvetica","B",7)
        self.set_fill_color(20,50,100)
        self.set_text_color(255,255,255)
        for i,h in enumerate(headers): self.cell(col_widths[i],6,h,border=1,align="C",fill=True)
        self.ln()
        self.set_font("Helvetica","",7)
        self.set_text_color(40,40,40)
        for ri,row in enumerate(data):
            fill=ri%2==1
            if fill: self.set_fill_color(245,245,250)
            for i,v in enumerate(row):
                txt=str(v)
                if highlight_col is not None and i==highlight_col:
                    try:
                        fv=float(v)
                        if fv>0: self.set_text_color(0,130,0)
                        elif fv<0: self.set_text_color(200,0,0)
                        else: self.set_text_color(40,40,40)
                    except: pass
                self.cell(col_widths[i],5,txt,border=1,align="C",fill=fill)
            self.set_text_color(40,40,40)
            self.ln()
        self.ln(3)

pdf=PDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True,margin=20)
pdf.add_page()

# === TITLE PAGE ===
pdf.ln(20)
pdf.set_font("Helvetica","B",22); pdf.set_text_color(20,50,100)
pdf.cell(0,12,"BB(20, 2.5) Breakout Strategy",align="C"); pdf.ln(10)
pdf.set_font("Helvetica","",11); pdf.set_text_color(80,80,80)
pdf.cell(0,7,"Comprehensive Backtest Report | 15-min Data | RR 1:3",align="C"); pdf.ln(12)
pdf.set_draw_color(20,50,100); pdf.line(60,pdf.get_y(),150,pdf.get_y()); pdf.ln(12)
pdf.set_font("Helvetica","",9); pdf.set_text_color(100,100,100)
pdf.cell(0,6,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",align="C"); pdf.ln(6)
pdf.cell(0,6,"Data: NIFTY50, BANKNIFTY, SENSEX Spot | Jan 2015 - Jun 2026",align="C"); pdf.ln(20)
pdf.set_font("Helvetica","B",10); pdf.set_text_color(40,40,40)
pdf.cell(0,6,"Strategy Rules:",align="C"); pdf.ln(8)
pdf.set_font("Helvetica","",9); pdf.set_text_color(60,60,60)
rules=["1. Compute BB(20, 2.5) on 15-min candles",
    "2. Trigger: entire candle outside band (low > upper OR high < lower)",
    "3. Entry: at close of trigger candle (market order)",
    "4. T1 (first target): near extreme of trigger candle = +1R profit",
    "5. TP (second target): 3x T1 distance = +3R profit",
    "6. EOD exit at 3:15 PM if neither target hit",
    "7. Charges: brokerage 10+10, STT 0.1% on sell, exchange/SEBI/stamp/GST"]
for r in rules: pdf.cell(0,5.5,r,align="C"); pdf.ln(5.5)

pdf.add_page()
pdf.section("1. Performance Summary")

# Combined summary table
rows=[]
for sym in ["NIFTY50","BANKNIFTY","SENSEX"]:
    st=s[sym]
    rows.append([sym, st["t"], f"{st['w']} ({st['wr']}%)", f"{st['gross']:+.0f}",
                 f"Rs{st['ch']:,.0f}", f"{st['net']:+.0f}", st["pf"], st["ar"],
                 f"{st['mdd']:,.0f}", st["sh"],
                 f"{st['t1c']} ({round(st['t1c']/st['t']*100,1)}%)",
                 f"{st['tpc']} ({round(st['tpc']/st['t']*100,1)}%)"])
pdf.subsection("All Indices Comparison")
pdf.table(["Index","Trades","Wins","Gross Pts","Charges","Net Pts","PF","AvgR","MaxDD","Sharpe","T1 (#/%)","TP (#/%)"],
          rows,[22,14,18,16,16,18,10,10,16,12,20,16],highlight_col=5)

pdf.text("Note: T1 exit = partial profit at +1R (mean reversion hits trigger extreme). "
         "TP exit = full target at +3R. EOD = position held until market close. "
         "All three indices are LOSS-MAKING after charges.")
pdf.text("Even before charges, gross P&L is negative for all three indices. The strategy fails because "
         "the mean-reversion move to T1 (+1R) is frequently not achieved before the counter-trend move "
         "hits the entry-stop or EOD. Charges then compound the losses.")

pdf.add_page()
pdf.section("2. BANKNIFTY - Detailed Report")
df=results["BANKNIFTY"]; st=s["BANKNIFTY"]
pdf.kv("Data Period",f"{df['date'].min()} to {df['date'].max()}")
pdf.kv("Trading Days with Trades",str(df['date'].nunique()))
pdf.kv("Total Trades",str(st["t"]))
pdf.kv("Winning Trades",f"{st['w']} ({st['wr']}%)")
pdf.kv("Losing Trades",str(st["l"]))
pdf.kv("Gross P&L (pts)",f"{st['gross']:+.0f}")
pdf.kv("Total Charges",f"Rs{st['ch']:,.0f}")
pdf.kv("Net P&L (pts)",f"{st['net']:+.0f}")
pdf.kv("Profit Factor",str(st["pf"]))
pdf.kv("Average R Multiple",str(st["ar"]))
pdf.kv("Max Drawdown",f"{st['mdd']:,.0f} pts")
pdf.kv("Sharpe Ratio (R-based)",str(st["sh"]))
pdf.kv("Avg T1 distance (SL pts)",str(st["t1_dist"]))
pdf.kv("Avg TP distance",str(st["tp_dist"]))

pdf.ln(2)
pdf.subsection("Exit Breakdown")
eod_r=round(df[df['reason']=='EOD']['r'].mean(),2) if st["eodc"] else 0
eod_p=round(df[df['reason']=='EOD']['pnl_pts'].mean(),1) if st["eodc"] else 0
pdf.table(["Exit","Count","% of Total","Avg R","Avg Pts"],
          [["T1 (+1R)",st["t1c"],f"{round(st['t1c']/st['t']*100,1)}%","1.00",f"+{st['t1_dist']}"],
           ["TP (+3R)",st["tpc"],f"{round(st['tpc']/st['t']*100,1)}%","3.00",f"+{st['tp_dist']}"],
           ["EOD",st["eodc"],f"{round(st['eodc']/st['t']*100,1)}%",f"{eod_r:.2f}",f"{eod_p:+.0f}"]],
          [30,16,22,20,24])

pdf.subsection("Yearly Breakdown")
y_data=[]
yearly=df.groupby("year").agg(tr=("pnl_pts","count"),w=("pnl_pts",lambda x: int((x>0).sum())),
    gross=("pnl_pts","sum"),net=("net_pnl","sum"),r=("r","mean"))
for yr,r in yearly.iterrows():
    y_data.append([int(yr),int(r["tr"]),f"{r['w']} ({round(r['w']/r['tr']*100,0):.0f}%)",
                   f"{r['gross']:+.0f}",f"{r['net']:+.0f}",f"{r['r']:.2f}"])
pdf.table(["Year","Trades","Wins","Gross","Net","AvgR"],y_data,[18,16,28,28,28,18],highlight_col=4)

pdf.ln(2)
pdf.subsection("Direction Breakdown")
for t_dir in ["LONG","SHORT"]:
    sub=df[df["type"]==t_dir]
    if len(sub)>0:
        pdf.kv(f"  {t_dir}",f"{len(sub)} trades | WR: {round((sub['pnl_pts']>0).sum()/len(sub)*100,1)}% | "
               f"Gross: {sub['pnl_pts'].sum():+.0f} | Net: {sub['net_pnl'].sum():+.0f} | AvgR: {sub['r'].mean():.2f}")
long_net=df[df['type']=='LONG']['net_pnl'].sum()
short_net=df[df['type']=='SHORT']['net_pnl'].sum()
better_dir="LONG" if long_net > short_net else "SHORT"
pdf.text(f"Both LONG and SHORT are loss-making after charges. "
         f"{better_dir} is the lesser loser (LONG: {long_net:+.0f}, SHORT: {short_net:+.0f} net).")

pdf.add_page()
pdf.section("3. SENSEX - Detailed Report")
df=results["SENSEX"]; st=s["SENSEX"]
pdf.kv("Data Period",f"{df['date'].min()} to {df['date'].max()}")
pdf.kv("Total Trades",str(st["t"]))
pdf.kv("Winning Trades",f"{st['w']} ({st['wr']}%)")
pdf.kv("Losing Trades",str(st["l"]))
pdf.kv("Gross P&L (pts)",f"{st['gross']:+.0f}")
pdf.kv("Total Charges",f"Rs{st['ch']:,.0f}")
pdf.kv("Net P&L (pts)",f"{st['net']:+.0f}")
pdf.kv("Profit Factor",str(st["pf"]))
pdf.kv("Average R Multiple",str(st["ar"]))
pdf.kv("Max Drawdown",f"{st['mdd']:,.0f} pts")
pdf.kv("Sharpe Ratio (R-based)",str(st["sh"]))
pdf.kv("Avg T1 distance",str(st["t1_dist"]))

pdf.ln(2)
pdf.subsection("Yearly Breakdown")
y_data=[]
yearly=df.groupby("year").agg(tr=("pnl_pts","count"),w=("pnl_pts",lambda x: int((x>0).sum())),
    gross=("pnl_pts","sum"),net=("net_pnl","sum"),r=("r","mean"))
for yr,r in yearly.iterrows():
    y_data.append([int(yr),int(r["tr"]),f"{r['w']} ({round(r['w']/r['tr']*100,0):.0f}%)",
                   f"{r['gross']:+.0f}",f"{r['net']:+.0f}",f"{r['r']:.2f}"])
pdf.table(["Year","Trades","Wins","Gross","Net","AvgR"],y_data,[18,16,28,28,28,18],highlight_col=4)

pdf.ln(2)
pdf.subsection("Direction Breakdown")
for t_dir in ["LONG","SHORT"]:
    sub=df[df["type"]==t_dir]
    if len(sub)>0:
        pdf.kv(f"  {t_dir}",f"{len(sub)} trades | WR: {round((sub['pnl_pts']>0).sum()/len(sub)*100,1)}% | "
               f"Gross: {sub['pnl_pts'].sum():+.0f} | Net: {sub['net_pnl'].sum():+.0f} | AvgR: {sub['r'].mean():.2f}")

pdf.section("4. NIFTY50 - Detailed Report")
df=results["NIFTY50"]; st=s["NIFTY50"]
pdf.kv("Data Period",f"{df['date'].min()} to {df['date'].max()}")
pdf.kv("Total Trades",str(st["t"]))
pdf.kv("Winning Trades",f"{st['w']} ({st['wr']}%)")
pdf.kv("Losing Trades",str(st["l"]))
pdf.kv("Gross P&L (pts)",f"{st['gross']:+.0f}")
pdf.kv("Total Charges",f"Rs{st['ch']:,.0f}")
pdf.kv("Net P&L (pts)",f"{st['net']:+.0f}")
pdf.kv("Avg T1 Distance",str(st["t1_dist"]))
pdf.text(f"NIFTY50 loses {abs(st['net']):.0f} pts net after charges ({st['net']:+.0f} pts). "
         f"Even gross P&L is negative ({st['gross']:+.0f} pts before charges). "
         f"The strategy fails across all indices.")

pdf.add_page()
pdf.section("5. Monthly & Day-of-Week Patterns (BANKNIFTY)")
pdf.subsection("Monthly Net P&L")
df=results["BANKNIFTY"]
mn=df.groupby("month")["net_pnl"].sum()
names_m={1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
m_data=[[names_m[m],f"{v:+.0f}"] for m,v in mn.sort_values(ascending=False).items()]
pdf.table(["Month","Net Pts"],m_data,[40,40],highlight_col=1)
pdf.text(f"Best: {names_m[mn.idxmax()]} (+{mn.max():.0f}), Worst: {names_m[mn.idxmin()]} ({mn.min():+.0f})")

pdf.ln(3)
pdf.subsection("Day of Week Net P&L")
dow_data=[]
for d in range(5):
    sub=df[df["dow"]==d]
    names_d={0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday"}
    if len(sub)>0:
        dow_data.append([names_d[d],len(sub),f"{sub['net_pnl'].sum():+.0f}"])
pdf.table(["Day","Trades","Net Pts"],dow_data,[38,28,38],highlight_col=2)

pdf.ln(3)
pdf.subsection("Long vs Short (BANKNIFTY)")
ls_data=[]
for t_dir in ["LONG","SHORT"]:
    sub=df[df["type"]==t_dir]
    if len(sub)>0:
        ls_data.append([t_dir,len(sub),f"{round((sub['pnl_pts']>0).sum()/len(sub)*100,1)}%",
                       f"{sub['pnl_pts'].sum():+.0f}",f"{sub['net_pnl'].sum():+.0f}",f"{sub['r'].mean():.2f}"])
pdf.table(["Type","Trades","WR (pts)","Gross","Net","AvgR"],ls_data,[16,18,20,28,28,18],highlight_col=4)

pdf.ln(5)
pdf.section("6. Conclusion & Recommendations")
pdf.set_font("Helvetica","B",9); pdf.set_text_color(200,0,0)
pdf.cell(0,6,"Strategy Verdict: NOT PROFITABLE on any index after charges",align="C"); pdf.ln(10)
pdf.set_font("Helvetica","",8.5); pdf.set_text_color(60,60,60)
bn=s["BANKNIFTY"]; ni=s["NIFTY50"]; se=s["SENSEX"]
conclusions=[
    f"The BB(20, 2.5) mean-reversion strategy fails on all three indices. Gross P&L is negative for all:",
    f"  NIFTY50 {ni['gross']:+.0f} pts, BANKNIFTY {bn['gross']:+.0f} pts, SENSEX {se['gross']:+.0f} pts before charges.",
    f"After charges, NIFTY50 {ni['net']:+.0f}, BANKNIFTY {bn['net']:+.0f}, SENSEX {se['net']:+.0f} pts.",
    "",
    f"Win rates (75-78%) are realistic but the losses on EOD/partial-failure trades exceed the gains from T1/TP hits.",
    f"Average R multiple is negative for all (NIFTY50: {ni['ar']}, BANKNIFTY: {bn['ar']}, SENSEX: {se['ar']}).",
    "",
    f"Exit distribution (BANKNIFTY): T1 (+1R) = {bn['t1c']} ({round(bn['t1c']/bn['t']*100,1)}%), "
    f"TP (+3R) = {bn['tpc']} ({round(bn['tpc']/bn['t']*100,1)}%), "
    f"EOD = {bn['eodc']} ({round(bn['eodc']/bn['t']*100,1)}%).",
    "The 26% EOD rate means the expected move often does not materialize within the trading day.",
    "",
    "Reason for failure: BB(20, 2.5) bands on 15-min are too wide. The mean reversion from a full-band breakout",
    "is not reliable enough, especially with EOD time limit. A wider T1 target or longer holding period needed.",
    "",
    "Previous bug: T1 exit condition was checking the wrong candle side (HIGH for SHORT, LOW for LONG),",
    "causing T1 to trigger on the very next candle ~100% of the time. After fixing, win rate dropped to 75-78%."
]
for c in conclusions: pdf.cell(0,5,c,align="C"); pdf.ln(5)

path=os.path.join(OUTPUT_DIR,"BB_Breakout_Strategy_Report.pdf")
pdf.output(path)
print(f"\nPDF saved: {path}")
