"""
utils/preprocessing.py
======================
Preprocessing data CSV sebelum klasifikasi & forecasting.
"""

from __future__ import annotations

import warnings
import pandas as pd
import numpy as np

_BULAN_ID: dict[str, str] = {
    "Mei": "May",
    "Agu": "Aug",
    "Okt": "Oct",
    "Des": "Dec",
}

_DATE_CANDIDATES  = ["ds", "date", "tanggal", "bulan", "month", "period", "time", "periode"]
_SALES_CANDIDATES = ["y", "sales", "qty", "penjualan", "demand", "volume", "jumlah", "amount"]


def _parse_date_id(value: str) -> pd.Timestamp:
    s = str(value).strip()
    for id_name, en_name in _BULAN_ID.items():
        s = s.replace(id_name, en_name)
    try:
        return pd.to_datetime(s, format="%b-%y")
    except ValueError:
        pass
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT


def normalize_to_ds_y(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    if "ds" not in df.columns:
        found_date = None
        for cand in _DATE_CANDIDATES:
            if cand in cols_lower:
                found_date = cols_lower[cand]
                break
        if found_date is None:
            raise ValueError(
                f"Kolom tanggal tidak ditemukan. Kolom yang dicari: {_DATE_CANDIDATES}. "
                f"Kolom tersedia: {list(df.columns)}"
            )
        df = df.rename(columns={found_date: "ds"})

    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = df["ds"].apply(_parse_date_id)

    if "y" not in df.columns:
        found_sales = None
        for cand in _SALES_CANDIDATES:
            if cand in cols_lower:
                found_sales = cols_lower[cand]
                break
        if found_sales is None:
            raise ValueError(
                f"Kolom penjualan tidak ditemukan. Kolom yang dicari: {_SALES_CANDIDATES}. "
                f"Kolom tersedia: {list(df.columns)}"
            )
        df = df.rename(columns={found_sales: "y"})

    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_to_ds_y(df)

    if "ds" not in df.columns:
        raise ValueError("Kolom 'ds' tidak ditemukan.")
    if "y" not in df.columns:
        raise ValueError("Kolom 'y' tidak ditemukan.")

    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = df["ds"].apply(_parse_date_id)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    n_before  = len(df)
    df        = df.dropna(subset=["ds", "y"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        warnings.warn(f"preprocess_data: {n_dropped} baris dibuang karena ds/y tidak valid.")

    if df.empty:
        raise ValueError("Tidak ada data valid setelah preprocessing.")

    if "Model" not in df.columns:
        df["Model"] = "Product"

    df["Model"] = df["Model"].astype(str).str.strip()
    df["ds"]    = df["ds"].dt.to_period("M").dt.to_timestamp()

    extra_cols = [c for c in df.columns if c not in {"ds", "y", "Model"}]

    if extra_cols:
        extra_df = df.groupby(["Model", "ds"])[extra_cols].first().reset_index()

    agg_df = df.groupby(["Model", "ds"])["y"].sum().reset_index()

    if extra_cols:
        agg_df = agg_df.merge(extra_df, on=["Model", "ds"], how="left")

    filled_parts = []
    for model_name, grp in agg_df.groupby("Model"):
        grp      = grp.sort_values("ds").set_index("ds")
        full_idx = pd.date_range(start=grp.index.min(), end=grp.index.max(), freq="MS")
        grp         = grp.reindex(full_idx)
        grp["y"]    = grp["y"].fillna(0)
        grp["Model"] = model_name
        for col in extra_cols:
            if col in grp.columns:
                grp[col] = grp[col].ffill().bfill()
        grp = grp.reset_index().rename(columns={"index": "ds"})
        filled_parts.append(grp)

    if not filled_parts:
        raise ValueError("Tidak ada data setelah resample.")

    result = pd.concat(filled_parts, ignore_index=True)
    result = result.sort_values(["Model", "ds"]).reset_index(drop=True)
    return result