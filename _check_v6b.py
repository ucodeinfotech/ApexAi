import pickle, sys, torch
BASE = r'C:\Users\pc\Downloads\stock hist data'
sys.path.insert(0, BASE)

# Define StockTransformer and PositionalEncoding to unpickle
import torch.nn as nn, math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class StockTransformer(nn.Module):
    def __init__(self, n_features, d_model=32, nhead=4, num_layers=2, seq_len=20):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*2, dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.decoder(x).squeeze(-1)

with open(BASE + r'\return_prediction_report_v6\results_v6.pkl', 'rb') as f:
    v6 = pickle.load(f)

print('Loaded successfully!')
print(f'Keys: {list(v6.keys())}')
print(f'Predictions: {len(v6["rd"]):,}')
print(f'Features: {v6["features"]}')

bt = v6['bt']
rd = v6['rd']

import numpy as np
STRATS = [('Top-1','t1_ret','t1_net','t1_to'),('Top-3','t3_ret','t3_net','t3_to'),
          ('Top-5','t5_ret','t5_net','t5_to'),('Top-10','t10_ret','t10_net','t10_to')]

print(f'\n{"Strategy":15s} {"Gross CAGR":>10s} {"Net CAGR":>10s} {"Sharpe":>7s}')
print('-'*45)
for sn, rc, nc, tc in STRATS:
    g = bt[rc].dropna()
    n = bt[nc].dropna()
    if len(g) < 10: continue
    gc = (1+g/100).prod()**(252/len(g))-1
    nac = (1+n/100).prod()**(252/len(n))-1
    sh = n.mean()/n.std()*np.sqrt(252)
    print(f'{sn:15s} {gc*100:>+9.1f}% {nac*100:>+9.1f}% {sh:>6.2f}')

print(f'\n{"Model":12s} {"Net CAGR":>10s} {"Sharpe":>7s} {"WinRate":>8s}')
print('-'*42)
for col in ['xgb','ranker','lgb','lgb_r','cb','rf','et','transformer','avg','stack']:
    nc = f'{col}_net'
    if nc not in bt.columns: continue
    n = bt[nc].dropna()
    if len(n) < 10: continue
    cagr = (1+n/100).prod()**(252/len(n))-1
    sh = n.mean()/n.std()*np.sqrt(252)
    wr = (n>0).mean()
    print(f'{col:12s} {cagr*100:>+9.1f}% {sh:>6.2f} {wr:>7.1%}')

print(f'\nDirectional Accuracy (Close-Close):')
for col in ['xgb','ranker','lgb','lgb_r','cb','rf','et','transformer','avg','stack']:
    if col not in rd.columns: continue
    da_cc = ((rd[col]>0)==(rd['act']>0)).mean()
    da_oc = ((rd[col]>0)==(rd['act_open']>0)).mean()
    print(f'  {col:12s} CC-DirAcc={da_cc:.1%}  OC-DirAcc={da_oc:.1%}')

rd['dt'] = pd.to_datetime(rd['dt'])
print(f'\nDate range: {rd["dt"].min()} to {rd["dt"].max()}')
print(f'Trading days: {rd["dt"].dt.normalize().nunique()}')
print(f'Symbols: {rd["sym"].nunique()}')
print(f'Total time: {v6["time"]/60:.1f} min')
