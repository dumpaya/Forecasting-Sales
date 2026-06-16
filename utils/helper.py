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