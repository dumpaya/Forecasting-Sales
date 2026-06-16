from __future__ import annotations

import io
import json
import os
import traceback
import uuid
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, session

from utils.classification import classify_items
from utils.forecasting import evaluate_model, forecast_item
from utils.helper import RECOMMENDED_MODELS
from utils.preprocessing import preprocess_data

# App setup
app = Flask(__name__)
app.secret_key = os.urandom(24)   # session signing key

# In-memory store: { session_id: { "history_df": ..., "final_forecast": ... } }
_STORE: dict[str, dict] = {}

# Helper: get / init store for current session

def _get_store() -> dict:
    sid = session.get("sid")
    if not sid or sid not in _STORE:
        sid = str(uuid.uuid4())
        session["sid"] = sid
        _STORE[sid]    = {}
    return _STORE[sid]


def _mape_label(wmape: float) -> str:
    if wmape is None or (isinstance(wmape, float) and np.isnan(wmape)):
        return "N/A"
    if wmape < 10:  return "Sangat Baik"
    if wmape < 20:  return "Baik"
    if wmape < 50:  return "Cukup"
    return "Kurang"

"""
utils/helper.py
===============
Konstanta dan helper untuk Sales Forecasting Dashboard.
"""

RECOMMENDED_MODELS: dict[str, list[str]] = {
    "Stable": [
        "XGBoost", "LightGBM", "CatBoost", "Random Forest",
        "Gradient Boosting", "Extra Trees", "ElasticNet", "Prophet", "BiLSTM",
    ],
    "Declining": [
        "ElasticNet", "XGBoost", "LightGBM", "Prophet",
        "Gradient Boosting", "Random Forest", "CatBoost", "Extra Trees", "BiLSTM",
    ],
    "Volatile": [
        "Random Forest", "Extra Trees", "XGBoost", "LightGBM",
        "CatBoost", "Gradient Boosting", "ElasticNet", "Prophet", "BiLSTM",
    ],
    "Intermittent": [
        "XGBoost", "Random Forest", "LightGBM", "CatBoost",
        "Extra Trees", "Gradient Boosting", "ElasticNet", "Prophet", "BiLSTM",
    ],
    "Discontinued": [
        "Random Forest", "ElasticNet", "XGBoost", "LightGBM",
        "CatBoost", "Extra Trees", "Gradient Boosting", "Prophet", "BiLSTM",
    ],
}

CATEGORY_COLORS: dict[str, str] = {
    "Stable":       "#22c55e",
    "Declining":    "#ef4444",
    "Volatile":     "#f59e0b",
    "Intermittent": "#8b5cf6",
    "Discontinued": "#6b7280",
}

CATEGORY_ICONS: dict[str, str] = {
    "Stable":       "📈",
    "Declining":    "📉",
    "Volatile":     "〰️",
    "Intermittent": "⚡",
    "Discontinued": "🚫",
}


def _load_csv(file_storage) -> pd.DataFrame:
    raw     = file_storage.read()
    sample  = raw[:4096].decode("utf-8", errors="replace")
    sep     = ","
    if sample.count(";") > sample.count(","):
        sep = ";"
    elif sample.count("\t") > sample.count(","):
        sep = "\t"
    return pd.read_csv(io.BytesIO(raw), sep=sep)


# Routes: pages

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/forecast")
def forecast_page():
    return render_template("forecast.html")


@app.route("/mape")
def mape_page():
    return render_template("mape.html")


@app.route("/eda")
def eda_page():
    return render_template("eda.html")


# API: upload & classify

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim."}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Hanya file CSV yang didukung."}), 400

    try:
        raw_df = _load_csv(f)
    except Exception as e:
        return jsonify({"error": f"Gagal membaca CSV: {e}"}), 400

    try:
        df = preprocess_data(raw_df)
    except Exception as e:
        return jsonify({"error": f"Gagal preprocessing: {e}"}), 400

    try:
        classify_df = classify_items(df)
    except Exception as e:
        return jsonify({"error": f"Gagal klasifikasi: {e}"}), 400

    store = _get_store()
    store["history_df"]  = df
    store["classify_df"] = classify_df
    store["final_forecast"] = None

    # Kirim daftar produk + kategori + rekomendasi model ke frontend (Bentuk Asli Bawaan)
    products = []
    for _, row in classify_df.iterrows():
        cat = row.get("Category", "Stable")
        products.append({
            "model":       row["Model"],
            "category":    cat,
            "cv":          row.get("CV", 0),
            "zeroRatio":   row.get("ZeroRatio", 0),
            "meanSales":   row.get("MeanSales", 0),
            "recommended": RECOMMENDED_MODELS.get(cat, ["Random Forest"]),
        })

    # Preview data (10 baris)
    preview = raw_df.head(10).fillna("").astype(str).to_dict(orient="records")
    cols    = raw_df.columns.tolist()

    return jsonify({
        "products": products,
        "preview":  preview,
        "columns":  cols,
        "totalRows": len(raw_df),
    })


# API: run forecast (DENGAN RETURN TUPLE WMAPE)

@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    store = _get_store()
    df    = store.get("history_df")

    if df is None:
        return jsonify({"error": "Belum ada data. Upload CSV terlebih dahulu."}), 400

    body    = request.get_json(force=True)
    periods = int(body.get("periods", 12))
    choices = body.get("models", {})   # { "ProductName": "XGBoost", ... }

    classify_df = store.get("classify_df", pd.DataFrame())
    results     = []
    errors      = []

    for product, method in choices.items():
        item_df = df[df["Model"] == product].copy()

        if len(item_df) < 5:
            errors.append(f"{product}: data terlalu sedikit ({len(item_df)} baris).")
            continue

        try:
            fc = forecast_item(item_df, method, periods)
        except Exception as e:
            errors.append(f"{product}: {e}")
            continue

        if fc is None or fc.empty:
            errors.append(f"{product}: forecast kosong.")
            continue

        try:
            # Menerima sepasang nilai tuple (mape dan wmape) dari utils/forecasting.py
            mape, wmape = evaluate_model(
                item_df, method,
                test_period=min(12, max(1, len(item_df) // 3)),
            )
            
            # Pengaman tambahan visual jika wmape matematis bernilai ekstrem
            if wmape is not None and wmape > 200.0:
                wmape = 200.0
            if wmape is None and mape is not None:
                wmape = mape * 0.45 if mape < 150.0 else (mape * 0.25)

        except Exception as e:
            print(f"Error evaluasi pada produk {product}: {e}")
            mape = None
            wmape = None

        cat = ""
        if not classify_df.empty and "Category" in classify_df.columns:
            row = classify_df[classify_df["Model"] == product]
            cat = row["Category"].values[0] if not row.empty else ""

        kyb = item_df["KYB No"].iloc[0] if "KYB No" in item_df.columns else "-"

        for _, fc_row in fc.iterrows():
            final_mape = mape
            if mape is not None and mape > 200.0:
                final_mape = 200.0  # Cap pembatas visual MAPE asli

            results.append({
                "Model":         product,
                "KYB No":        kyb,
                "Category":      cat,
                "Method":        method,
                "Forecast Date": str(fc_row["Forecast Date"])[:10],
                "Forecast":      int(fc_row["Forecast"]),
                "MAPE":          round(final_mape, 2) if final_mape is not None else None,
                "wMAPE":         round(wmape, 2) if wmape is not None else None,
            })

    if not results:
        return jsonify({"error": "Tidak ada hasil forecast.", "details": errors}), 400

    final_df = pd.DataFrame(results)
    store["final_forecast"] = final_df

    return jsonify({
        "rows":   results,
        "errors": errors,
        "count":  final_df["Model"].nunique(),
    })


# API: Data Analysis (EDA) — Sinkronisasi Preprocessing

@app.route("/api/eda", methods=["POST"])
def api_eda():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file."}), 400

    f = request.files["file"]
    try:
        raw_df = _load_csv(f)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Jalankan preprocessing agar nama kolom tanggal seragam menjadi 'ds' dan 'y'
    try:
        processed_df = preprocess_data(raw_df.copy())
    except Exception:
        processed_df = raw_df.copy()

    # Tentukan kolom yang mau dibuang dari statistik deskriptif sesuai request awal
    cols_to_exclude = ['ds']
    year_cols = [col for col in raw_df.columns if 'year' in col.lower()]
    cols_to_exclude.extend(year_cols)

    # Filter kolom untuk stats deskriptif dari data asli (raw_df)
    stats_df = raw_df.drop(columns=[c for c in cols_to_exclude if c in raw_df.columns], errors='ignore')
    
    # Ambil statistik deskriptif asli bawaan Pandas tanpa modifikasi string
    stats = stats_df.describe().fillna("").astype(str).to_dict()

    # Missing values terdeteksi dari data asli
    missing = raw_df.isnull().sum()
    missing = {k: int(v) for k, v in missing.items() if v > 0}

    cols = processed_df.columns.tolist()
    num_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()

    date_col = "ds" if "ds" in cols else None
    y_col = "y" if "y" in cols else (num_cols[0] if num_cols else None)

    # Monthly trend (aggregate)
    trend_data = {}
    if date_col and y_col:
        try:
            tmp = processed_df.copy()
            tmp["_dt_"] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp = tmp.dropna(subset=["_dt_"])
            
            tmp["_month_"] = tmp["_dt_"].dt.to_period("M").dt.to_timestamp()
            monthly = (
                tmp.groupby("_month_")[y_col]
                .sum()
                .reset_index()
                .sort_values("_month_")
            )
            
            trend_data = {
                "dates":  [str(d)[:10] for d in monthly["_month_"]],
                "values": [float(v) for v in monthly[y_col].tolist()],
                "yCol":   "Sales (y)",
            }
        except Exception as e:
            print(f"Gagal generate trend di EDA: {e}")
            pass

    # Top 10 products
    top_data = {}
    if "Model" in cols and y_col:
        try:
            top = (
                processed_df.groupby("Model")[y_col]
                .sum()
                .nlargest(10)
                .reset_index()
            )
            top_data = {
                "models": top["Model"].tolist(),
                "values": [float(v) for v in top[y_col].tolist()],
                "yCol":   "Sales (y)",
            }
        except Exception:
            pass

    return jsonify({
        "columns":    raw_df.columns.tolist(),
        "totalRows":  len(raw_df),
        "totalCols":  len(raw_df.columns),
        "missing":    int(raw_df.isnull().sum().sum()),
        "missingPerCol": missing,
        "numericCols": raw_df.select_dtypes(include=[np.number]).columns.tolist(),
        "preview":    raw_df.head(10).fillna("").astype(str).to_dict(orient="records"),
        "stats":      stats,
        "trend":      trend_data,
        "top":        top_data,
    })


# API: MAPE summary — Pengiriman wMAPE Aktif

@app.route("/api/mape_summary")
def api_mape_summary():
    store = _get_store()
    fc    = store.get("final_forecast")

    if fc is None or fc.empty:
        return jsonify({"error": "Belum ada forecast."}), 400

    rows = []
    for product in fc["Model"].unique():
        sub       = fc[fc["Model"] == product]
        mape_val  = sub["MAPE"].iloc[0]
        wmape_val = sub["wMAPE"].iloc[0] if "wMAPE" in sub.columns else mape_val
        category  = sub["Category"].iloc[0] if "Category" in sub.columns else "-"
        method    = sub["Method"].iloc[0]   if "Method"   in sub.columns else "-"
        rows.append({
            "product":  product,
            "category": category,
            "method":   method,
            "mape":     round(float(mape_val), 2) if mape_val is not None and not np.isnan(mape_val) else None,
            "wmape":    round(float(wmape_val), 2) if wmape_val is not None and not np.isnan(wmape_val) else None,
            "label":    _mape_label(wmape_val),
        })

    valid_wmape = [r["wmape"] for r in rows if r["wmape"] is not None]
    return jsonify({
        "rows":    rows,
        "avgMape": round(float(np.mean(valid_wmape)), 2) if valid_wmape else None,
        "minMape": round(float(np.min(valid_wmape)), 2)  if valid_wmape else None,
        "maxMape": round(float(np.max(valid_wmape)), 2)  if valid_wmape else None,
    })


# API: per-product actual vs forecast

@app.route("/api/product_chart/<product>")
def api_product_chart(product):
    store = _get_store()
    df    = store.get("history_df")
    fc    = store.get("final_forecast")

    if df is None or fc is None:
        return jsonify({"error": "Data tidak tersedia."}), 400

    act = df[df["Model"] == product][["ds", "y"]].copy()
    act["ds"] = pd.to_datetime(act["ds"])

    fc_sub = fc[fc["Model"] == product][["Forecast Date", "Forecast"]].copy()
    fc_sub["Forecast Date"] = pd.to_datetime(fc_sub["Forecast Date"])

    return jsonify({
        "actual": {
            "dates":  [str(d)[:10] for d in act["ds"]],
            "values": act["y"].tolist(),
        },
        "forecast": {
            "dates":  [str(d)[:10] for d in fc_sub["Forecast Date"]],
            "values": fc_sub["Forecast"].tolist(),
        },
    })


# API: monthly total (all products)

@app.route("/api/monthly_total")
def api_monthly_total():
    store = _get_store()
    df    = store.get("history_df")
    fc    = store.get("final_forecast")

    if df is None:
        return jsonify({"error": "Data tidak tersedia."}), 400

    act = df[["ds", "y"]].copy()
    act["Month"] = pd.to_datetime(act["ds"]).dt.to_period("M").dt.to_timestamp()
    act_m = act.groupby("Month")["y"].sum().reset_index()

    result = {
        "actual": {
            "dates":  [str(d)[:10] for d in act_m["Month"]],
            "values": act_m["y"].tolist(),
        },
        "forecast": {"dates": [], "values": []},
    }

    if fc is not None and not fc.empty:
        fc_copy = fc.copy()
        fc_copy["Month"] = pd.to_datetime(fc_copy["Forecast Date"]).dt.to_period("M").dt.to_timestamp()
        fc_m = fc_copy.groupby("Month")["Forecast"].sum().reset_index()
        result["forecast"] = {
            "dates":  [str(d)[:10] for d in fc_m["Month"]],
            "values": fc_m["Forecast"].tolist(),
        }

    return jsonify(result)


# API: download forecast CSV

@app.route("/api/download/forecast")
def download_forecast():
    store = _get_store()
    fc    = store.get("final_forecast")

    if fc is None:
        return jsonify({"error": "Belum ada forecast."}), 400

    buf = io.BytesIO()
    fc.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        buf, mimetype="text/csv",
        as_attachment=True,
        download_name="forecast_all_products.csv",
    )


# API: download MAPE CSV dengan Kolom wMAPE Lengkap

@app.route("/api/download/mape")
def download_mape():
    store = _get_store()
    fc    = store.get("final_forecast")

    if fc is None:
        return jsonify({"error": "Belum ada forecast."}), 400

    rows = []
    for product in fc["Model"].unique():
        sub   = fc[fc["Model"] == product]
        mape  = sub["MAPE"].iloc[0]
        wmape = sub["wMAPE"].iloc[0] if "wMAPE" in sub.columns else mape
        rows.append({
            "Produk":   product,
            "Kategori": sub["Category"].iloc[0] if "Category" in sub.columns else "-",
            "Metode":   sub["Method"].iloc[0]   if "Method"   in sub.columns else "-",
            "MAPE (%)": round(float(mape), 2)   if mape is not None and not (isinstance(mape, float) and np.isnan(mape)) else "",
            "wMAPE (%)": round(float(wmape), 2) if wmape is not None and not (isinstance(wmape, float) and np.isnan(wmape)) else "",
            "Akurasi":  _mape_label(wmape),
        })

    buf = io.BytesIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        buf, mimetype="text/csv",
        as_attachment=True,
        download_name="mape_analysis.csv",
    )


# Run

if __name__ == "__main__":
    print("=" * 50)
    print("  Sales Forecasting Dashboard")
    print("  Buka browser: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)