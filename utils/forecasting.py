"""
utils/forecasting.py
====================
Modul forecasting ML & deep learning. Tanpa dependensi Streamlit.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from prophet import Prophet

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping


# ── Feature Engineering ───────────────────────────────────────────────────────

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    df["month"]          = df["ds"].dt.month
    df["year"]           = df["ds"].dt.year
    df["quarter"]        = df["ds"].dt.quarter
    df["lag_1"]          = df["y"].shift(1)
    df["lag_2"]          = df["y"].shift(2)
    df["lag_3"]          = df["y"].shift(3)
    df["lag_6"]          = df["y"].shift(6)
    df["lag_12"]         = df["y"].shift(12)
    df["rolling_mean_3"] = df["y"].rolling(3).mean()
    df["rolling_mean_6"] = df["y"].rolling(6).mean()
    df["rolling_std_3"]  = df["y"].rolling(3).std()

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


FEATURE_COLS = [
    "month", "year", "quarter",
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "rolling_mean_3", "rolling_mean_6", "rolling_std_3",
]


# ── Model Factory ─────────────────────────────────────────────────────────────

def get_model(method: str):
    models = {
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=42, verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1,
        ),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "CatBoost": CatBoostRegressor(
            iterations=200, learning_rate=0.05, depth=6,
            verbose=0, random_seed=42,
        ),
        "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42,
        ),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
    }
    return models.get(method, RandomForestRegressor(random_state=42))


# ── Prophet ───────────────────────────────────────────────────────────────────

def prophet_forecast(item_df: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
    prophet_df = item_df[["ds", "y"]].copy()
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df["y"]  = prophet_df["y"].clip(lower=0)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.8,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_df)

    future   = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)

    result = forecast.tail(periods)[["ds", "yhat"]].copy()
    result.columns = ["Forecast Date", "Forecast"]
    result["Forecast"] = np.clip(result["Forecast"], 0, None).round().astype(int)
    result["Forecast Date"] = pd.to_datetime(result["Forecast Date"])
    return result


# ── BiLSTM ────────────────────────────────────────────────────────────────────

def bilstm_forecast(item_df: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
    values = item_df["y"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    seq_length = min(12, len(scaled) // 2)
    if len(scaled) <= seq_length + 1:
        return pd.DataFrame({"Forecast Date": [], "Forecast": []})

    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i: i + seq_length])
        y.append(scaled[i + seq_length])

    X = np.array(X).reshape(-1, seq_length, 1)
    y = np.array(y)

    model = Sequential([
        Bidirectional(LSTM(64, activation="relu"), input_shape=(seq_length, 1)),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    early_stop = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=8, verbose=0, callbacks=[early_stop])

    current_batch = scaled[-seq_length:].reshape(1, seq_length, 1)
    predictions   = []
    for _ in range(periods):
        pred = model.predict(current_batch, verbose=0)[0]
        predictions.append(pred)
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    predictions = np.clip(predictions, 0, None).round().astype(int)

    future_dates = pd.date_range(
        start=pd.to_datetime(item_df["ds"].max()) + pd.DateOffset(months=1),
        periods=periods, freq="MS",
    )
    return pd.DataFrame({"Forecast Date": future_dates, "Forecast": predictions})


# ── ML Recursive ─────────────────────────────────────────────────────────────

def ml_forecast(item_df: pd.DataFrame, method: str, periods: int = 12) -> pd.DataFrame:
    feat_df            = create_features(item_df)
    available_features = [c for c in FEATURE_COLS if c in feat_df.columns]

    if feat_df.empty or len(feat_df) < 3:
        return pd.DataFrame({"Forecast Date": [], "Forecast": []})

    X     = feat_df[available_features].fillna(0)
    y     = feat_df["y"]
    model = get_model(method)
    model.fit(X, y)

    temp_df   = item_df[["ds", "y"]].copy()
    temp_df["ds"] = pd.to_datetime(temp_df["ds"])
    temp_df   = temp_df.sort_values("ds").reset_index(drop=True)
    last_date = temp_df["ds"].max()
    results   = []

    for i in range(periods):
        future_date = last_date + pd.DateOffset(months=i + 1)
        n = len(temp_df)

        row = {
            "month":          future_date.month,
            "year":           future_date.year,
            "quarter":        future_date.quarter,
            "lag_1":          temp_df["y"].iloc[-1]  if n >= 1  else 0,
            "lag_2":          temp_df["y"].iloc[-2]  if n >= 2  else 0,
            "lag_3":          temp_df["y"].iloc[-3]  if n >= 3  else 0,
            "lag_6":          temp_df["y"].iloc[-6]  if n >= 6  else 0,
            "lag_12":         temp_df["y"].iloc[-12] if n >= 12 else 0,
            "rolling_mean_3": temp_df["y"].tail(3).mean() if n >= 1 else 0,
            "rolling_mean_6": temp_df["y"].tail(6).mean() if n >= 1 else 0,
            "rolling_std_3":  temp_df["y"].tail(3).std()  if n >= 2 else 0,
        }

        X_future = pd.DataFrame([{k: row[k] for k in available_features}]).fillna(0)
        pred     = max(float(model.predict(X_future)[0]), 0)

        results.append({"Forecast Date": future_date, "Forecast": round(pred)})
        temp_df = pd.concat(
            [temp_df, pd.DataFrame({"ds": [future_date], "y": [pred]})],
            ignore_index=True,
        )

    forecast_df = pd.DataFrame(results)
    forecast_df["Forecast Date"] = pd.to_datetime(forecast_df["Forecast Date"])
    forecast_df["Forecast"]      = forecast_df["Forecast"].astype(int)
    return forecast_df


# ── Entry Point ───────────────────────────────────────────────────────────────

def forecast_item(item_df: pd.DataFrame, method: str, periods: int = 12) -> pd.DataFrame:
    item_df = item_df[["ds", "y"]].copy()
    item_df["ds"] = pd.to_datetime(item_df["ds"])
    item_df["y"]  = pd.to_numeric(item_df["y"], errors="coerce").fillna(0)
    item_df = item_df.sort_values("ds").reset_index(drop=True)

    if len(item_df) < 3:
        return pd.DataFrame({"Forecast Date": [], "Forecast": []})

    if method == "Prophet":
        return prophet_forecast(item_df, periods)
    if method == "BiLSTM":
        return bilstm_forecast(item_df, periods)
    return ml_forecast(item_df, method, periods)


# ── Evaluasi & Metrik Kesalahan (Revisi) ──────────────────────────────────────

def calculate_mape(actual: np.ndarray, forecast: np.ndarray) -> float | None:
    actual   = np.array(actual,   dtype=float)
    forecast = np.array(forecast, dtype=float)
    mask     = actual != 0
    if mask.sum() == 0:
        return None
    mape = np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100
    return round(float(mape), 2)


def calculate_wmape(actual: np.ndarray, forecast: np.ndarray) -> float | None:
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)
    total_actual = np.sum(actual)
    if total_actual == 0:
        return None
    wmape = (np.sum(np.abs(actual - forecast)) / total_actual) * 100
    return round(float(wmape), 2)


def evaluate_model(item_df: pd.DataFrame, method: str, test_period: int = 12) -> tuple[float | None, float | None]:
    """Mengevaluasi model menggunakan Backtesting dan mengembalikan (MAPE, wMAPE) sekaligus"""
    item_df = item_df[["ds", "y"]].copy()
    item_df["ds"] = pd.to_datetime(item_df["ds"])
    item_df = item_df.sort_values("ds").reset_index(drop=True)

    min_train = max(5, test_period)
    if len(item_df) <= min_train + test_period:
        return None, None

    train = item_df.iloc[:-test_period].copy()
    test  = item_df.iloc[-test_period:].copy()

    try:
        prediction = forecast_item(train, method, periods=test_period)
    except Exception:
        return None, None

    if prediction is None or prediction.empty:
        return None, None

    actual   = test["y"].values
    forecast = prediction["Forecast"].values
    min_len  = min(len(actual), len(forecast))
    if min_len == 0:
        return None, None

    # Kalkulasi kedua metrik secara simultan
    mape_val = calculate_mape(actual[:min_len], forecast[:min_len])
    wmape_val = calculate_wmape(actual[:min_len], forecast[:min_len])

    return mape_val, wmape_val