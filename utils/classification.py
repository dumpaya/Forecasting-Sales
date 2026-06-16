"""
utils/classification.py
=======================
Klasifikasi produk berdasarkan pola historis penjualan.
(Improved version - robust against high MAPE cases)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ZERO_RATIO_THRESHOLD = 0.50     # dinaikkan biar ga over-detect intermittent
CV_STABLE_THRESHOLD  = 0.50
DECLINE_THRESHOLD    = -0.10
DISCONTINUED_MONTHS  = 3
LOW_MEAN_THRESHOLD   = 5        # tambahan: untuk hindari noise


def _classify_single(grp: pd.DataFrame) -> str:
    y = grp["y"].fillna(0).values

    if len(y) == 0:
        return "Discontinued"

    mean_val = y.mean()

    # 🔴 1. Discontinued check (PRIORITAS)
    recent = y[-DISCONTINUED_MONTHS:] if len(y) >= DISCONTINUED_MONTHS else y
    if recent.sum() == 0:
        return "Discontinued"

    # 🔴 2. Zero-heavy (Intermittent)
    zero_ratio = (y == 0).sum() / len(y)
    if zero_ratio > ZERO_RATIO_THRESHOLD:
        return "Intermittent"

    # 🔴 3. Low demand (hindari salah classify volatile)
    if mean_val < LOW_MEAN_THRESHOLD:
        return "Intermittent"

    std_val = y.std()
    cv      = std_val / mean_val if mean_val > 0 else 0

    # 🔴 4. Trend detection (Declining)
    if len(y) >= 6:  # diperketat biar lebih stabil
        t = np.arange(len(y))
        slope = np.polyfit(t, y, 1)[0]

        # normalisasi lebih stabil
        norm_slope = slope / (mean_val + 1e-6)

        if norm_slope < DECLINE_THRESHOLD:
            return "Declining"

    # 🔴 5. Stable vs Volatile
    if cv <= CV_STABLE_THRESHOLD:
        return "Stable"

    return "Volatile"


def classify_items(df: pd.DataFrame) -> pd.DataFrame:
    if "Model" not in df.columns:
        raise ValueError("Kolom 'Model' tidak ditemukan.")
    if "y" not in df.columns:
        raise ValueError("Kolom 'y' tidak ditemukan.")

    records = []

    for model_name, grp in df.groupby("Model"):
        grp = grp.sort_values("ds") if "ds" in grp.columns else grp
        y   = grp["y"].fillna(0).values

        category   = _classify_single(grp)
        mean_val   = float(y.mean())    if len(y) > 0 else 0.0
        std_val    = float(y.std())     if len(y) > 0 else 0.0
        cv         = std_val / mean_val if mean_val > 0 else 0.0
        zero_ratio = float((y == 0).sum() / len(y)) if len(y) > 0 else 1.0

        records.append({
            "Model":      model_name,
            "Category":   category,
            "DataPoints": len(y),
            "MeanSales":  round(mean_val, 2),
            "CV":         round(cv, 3),
            "ZeroRatio":  round(zero_ratio, 3),
        })

    return pd.DataFrame(records)