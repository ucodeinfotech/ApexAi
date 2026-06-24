"""
ML Models — XGBoost classifiers/regressors and LSTM sequence models.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
import joblib

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(r"C:\Users\pc\Downloads\stock hist data")
DB_PATH = BASE_DIR / "warehouse" / "market_data.duckdb"
MODEL_DIR = BASE_DIR / "warehouse" / "models"
MODEL_DIR.mkdir(exist_ok=True)


FEATURE_COLS = [
    "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
    "ema_5", "ema_10", "ema_20", "ema_50", "ema_200",
    "rsi_7", "rsi_14", "rsi_21",
    "macd_line", "macd_signal", "macd_hist",
    "adx", "plus_di", "minus_di",
    "atr_7", "atr_14", "atr_21",
    "bb_pct_b", "bb_width", "kc_width", "dc_width",
    "obv", "vol_ratio_5", "vol_ratio_10", "vol_ratio_20", "cmf",
    "stoch_k", "stoch_d", "williams_r", "mfi", "uo",
    "cci", "trix", "roc_5", "roc_10", "roc_20",
    "zscore_20", "skew_20", "kurt_20", "hv_10", "hv_20", "hv_30",
    "eom", "fi", "vpt", "ad_line",
]


class HighGainerModel:
    def __init__(self, model_type="xgb_classifier"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None

    def prepare_data(self, df, target_col="target", test_size=0.2):
        """Split into train/test with time-based split."""
        df = df.dropna(subset=[target_col]).copy()
        df = df.sort_values("datetime")

        # Select features
        avail = [c for c in FEATURE_COLS if c in df.columns]
        self.feature_cols = avail
        X = df[avail].fillna(0).values
        y = df[target_col].values

        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale
        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        return X_train_s, X_test_s, y_train, y_test

    def train_xgb_classifier(self, X_train, y_train, X_test, y_test):
        """Train XGBoost classifier for high-gainer prediction."""
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        self.model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=42,
            eval_metric="auc", early_stopping_rounds=30,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        return self._eval_classifier(X_test, y_test)

    def train_xgb_regressor(self, X_train, y_train, X_test, y_test):
        """Train XGBoost regressor for forward return prediction."""
        self.model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=42, eval_metric="mae", early_stopping_rounds=30,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        return self._eval_regressor(X_test, y_test)

    def _eval_classifier(self, X_test, y_test):
        preds = (self.model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
        probs = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        return {"auc": auc, "report": classification_report(y_test, preds, output_dict=True)}

    def _eval_regressor(self, X_test, y_test):
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        return {"mae": mae, "corr": float(np.corrcoef(y_test, preds)[0, 1])}

    def save(self, name):
        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump({"model": self.model, "scaler": self.scaler,
                      "feature_cols": self.feature_cols, "model_type": self.model_type}, path)
        return path

    def load(self, name):
        path = MODEL_DIR / f"{name}.joblib"
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_cols = data["feature_cols"]
        self.model_type = data["model_type"]


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        return self.classifier(out).squeeze(-1)


def train_lstm(df, target_col="target", seq_len=20, epochs=50, lr=0.001):
    """Train LSTM for sequence-based high-gainer prediction."""
    df = df.dropna(subset=[target_col]).sort_values(["symbol", "datetime"]).copy()
    avail = [c for c in FEATURE_COLS if c in df.columns]
    X_all = df[avail].fillna(0).values
    y_all = df[target_col].values

    # Create sequences
    X_seq, y_seq = [], []
    for i in range(len(X_all) - seq_len):
        X_seq.append(X_all[i:i + seq_len])
        y_seq.append(y_all[i + seq_len])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Scale per feature
    scaler = StandardScaler()
    orig_shape = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(orig_shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Dataset
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_dim=X_train.shape[-1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

    # Evaluate
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            out = torch.sigmoid(model(Xb.to(device))).cpu()
            preds.extend(out.numpy())
            targets.extend(yb.numpy())

    auc = roc_auc_score(targets, preds) if len(set(targets)) > 1 else 0.5
    return model, {"auc": auc, "train_loss": train_losses}
