"""
Streamlit Dashboard — Industrial Predictive Maintenance.

Run:
    streamlit run dashboard/app.py

Features:
  - Live sensor visualisation (rolling trend charts)
  - RUL gauge per engine
  - Anomaly timeline
  - Model metrics summary
"""

import json
import time
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---- Page config ----
st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = Path("models")
DATA_DIR = Path("data/raw")

# ---- Helpers ----

@st.cache_resource
def load_preprocessor():
    candidates = list(MODELS_DIR.glob("preprocessor_*.pkl"))
    if not candidates:
        return None
    with open(candidates[0], "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_rul_model():
    from src.models.rul import XGBoostRUL, LSTMRul
    xgb = list(MODELS_DIR.glob("rul_xgboost_*.joblib"))
    lstm = list(MODELS_DIR.glob("rul_lstm_*.pt"))
    if xgb:
        return XGBoostRUL.load(xgb[0])
    if lstm:
        return LSTMRul.load(lstm[0])
    return None

@st.cache_resource
def load_anomaly_model():
    from src.models.anomaly import AnomalyDetector
    cands = list(MODELS_DIR.glob("anomaly_*.joblib"))
    if cands:
        return AnomalyDetector.load(cands[0])
    return None

@st.cache_data
def load_metrics():
    p = MODELS_DIR / "metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

@st.cache_data
def load_sample_data():
    """Load real data if available, else generate synthetic."""
    from src.data.loader import load_cmapss, generate_synthetic_cmapss
    try:
        train_df, test_df, test_rul = load_cmapss(data_dir=DATA_DIR, subset="FD001")
        return train_df, test_df, test_rul, False
    except FileNotFoundError:
        train_df, test_df, test_rul = generate_synthetic_cmapss(n_engines=50, seed=42)
        return train_df, test_df, test_rul, True


def rul_color(rul: float) -> str:
    if rul > 80:
        return "#10b981"  # green
    if rul > 30:
        return "#f59e0b"  # amber
    return "#ef4444"      # red


def health_badge(rul: float) -> str:
    if rul > 80:
        return "🟢 HEALTHY"
    if rul > 30:
        return "🟡 WARNING"
    return "🔴 CRITICAL"


# ---- Sidebar ----
st.sidebar.image("https://img.icons8.com/fluency/96/000000/settings.png", width=64)
st.sidebar.title("⚙️ PM Dashboard")
st.sidebar.markdown("**Industrial AI — Turbofan Engine Health**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Fleet Overview", "🔍 Engine Deep-Dive", "🚨 Anomaly Monitor", "📈 Model Metrics"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data:** [NASA C-MAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)  \n"
    "**Code:** [github.com/bishoyad](https://github.com/bishoyad/industrial-predictive-maintenance)"
)

# ---- Load everything ----
train_df, test_df, test_rul, is_synthetic = load_sample_data()
preprocessor = load_preprocessor()
rul_model = load_rul_model()
anomaly_model = load_anomaly_model()
metrics = load_metrics()

models_ready = preprocessor is not None and rul_model is not None

if is_synthetic:
    st.warning(
        "⚠️ Using **synthetic data** (NASA C-MAPSS files not found in `data/raw/`).  "
        "Download the real dataset from [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) "
        "and place the `.txt` files in `data/raw/` for real results."
    )

# ===========================================================================
# PAGE 1 — Fleet Overview
# ===========================================================================
if page == "📊 Fleet Overview":
    st.title("📊 Fleet Health Overview")
    st.caption("Predicted Remaining Useful Life (RUL) across all monitored engines.")

    if not models_ready:
        st.error(
            "Models not found. Train first:\n```\npython -m src.models.trainer --synthetic\n```"
        )
        st.stop()

    # Predict RUL for all test engines
    with st.spinner("Running predictions …"):
        test_proc = preprocessor.transform(test_df)
        X_last = preprocessor.build_last_sequences(test_proc)
        preds = rul_model.predict(X_last)

    unit_ids = test_df["unit_id"].unique()[:len(preds)]
    result_df = pd.DataFrame({
        "Engine": [f"Engine {uid:03d}" for uid in unit_ids],
        "Predicted RUL": np.round(preds, 1),
        "True RUL": test_rul.values[:len(preds)],
        "Status": [health_badge(r) for r in preds],
    })

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏭 Total Engines", len(result_df))
    col2.metric("🟢 Healthy", (preds > 80).sum())
    col3.metric("🟡 Warning", ((preds > 30) & (preds <= 80)).sum())
    col4.metric("🔴 Critical", (preds <= 30).sum())

    st.markdown("---")

    # Fleet table
    st.subheader("Engine Fleet Status")
    st.dataframe(
        result_df.style.applymap(
            lambda v: f"color: {rul_color(float(v.split()[0])) if v[0].isdigit() else 'white'}",
            subset=["Predicted RUL"]
        ),
        use_container_width=True,
        height=400,
    )

    # RUL distribution
    st.subheader("RUL Distribution")
    hist_df = pd.DataFrame({"Predicted RUL": preds, "True RUL": test_rul.values[:len(preds)]})
    st.bar_chart(hist_df.describe().loc[["min", "25%", "50%", "75%", "max"]].T)


# ===========================================================================
# PAGE 2 — Engine Deep-Dive
# ===========================================================================
elif page == "🔍 Engine Deep-Dive":
    st.title("🔍 Engine Sensor Deep-Dive")

    engine_ids = sorted(train_df["unit_id"].unique())
    selected_id = st.selectbox("Select Engine Unit", engine_ids, index=0)
    engine_df = train_df[train_df["unit_id"] == selected_id].sort_values("cycle")

    max_cycle = engine_df["cycle"].max()
    true_rul_col = engine_df["rul"].iloc[-1] if "rul" in engine_df.columns else "N/A"

    col1, col2, col3 = st.columns(3)
    col1.metric("Engine ID", f"#{selected_id:03d}")
    col2.metric("Total Cycles", max_cycle)
    col3.metric("Final True RUL", true_rul_col)

    st.markdown("---")
    st.subheader("Sensor Readings Over Time")

    sensor_cols = [c for c in engine_df.columns if c.startswith("sensor_") and "_mean" not in c and "_std" not in c]
    selected_sensors = st.multiselect(
        "Select Sensors to Plot",
        sensor_cols,
        default=sensor_cols[:4],
    )

    if selected_sensors:
        chart_df = engine_df[["cycle"] + selected_sensors].set_index("cycle")
        st.line_chart(chart_df, height=350)
    else:
        st.info("Select at least one sensor above.")

    st.markdown("---")
    st.subheader("RUL Trajectory")
    if "rul" in engine_df.columns:
        rul_df = engine_df[["cycle", "rul"]].set_index("cycle")
        st.area_chart(rul_df, height=250, color="#10b981")


# ===========================================================================
# PAGE 3 — Anomaly Monitor
# ===========================================================================
elif page == "🚨 Anomaly Monitor":
    st.title("🚨 Real-Time Anomaly Monitor")

    if anomaly_model is None or preprocessor is None:
        st.error("Anomaly model not loaded. Train first: `python -m src.models.trainer --synthetic`")
        st.stop()

    engine_ids = sorted(train_df["unit_id"].unique())
    selected_id = st.selectbox("Monitor Engine", engine_ids, index=0)
    engine_df = train_df[train_df["unit_id"] == selected_id].sort_values("cycle").copy()

    with st.spinner("Detecting anomalies …"):
        engine_proc = preprocessor.transform(engine_df)
        X_all = preprocessor.build_sequences(engine_proc, target_col=None)[0] if hasattr(preprocessor, "build_sequences") else None

        if X_all is not None and len(X_all) > 0:
            scores = anomaly_model.anomaly_score(X_all)
            flags = anomaly_model.predict(X_all)
            # Align with cycles
            window = preprocessor.window
            cycles = engine_df["cycle"].values[window - 1:]
            score_df = pd.DataFrame({
                "cycle": cycles[:len(scores)],
                "anomaly_score": scores[:len(cycles)],
                "is_anomaly": flags[:len(cycles)],
            })
        else:
            score_df = pd.DataFrame(columns=["cycle", "anomaly_score", "is_anomaly"])

    col1, col2 = st.columns(2)
    if len(score_df) > 0:
        col1.metric("Anomalous Cycles", int(score_df["is_anomaly"].sum()))
        col2.metric("Max Anomaly Score", round(float(score_df["anomaly_score"].max()), 3))

        st.subheader("Anomaly Score Timeline")
        st.line_chart(score_df.set_index("cycle")["anomaly_score"], height=280)

        anomalous_rows = score_df[score_df["is_anomaly"] == 1]
        if len(anomalous_rows) > 0:
            st.warning(f"⚠️ {len(anomalous_rows)} anomalous cycles detected!")
            st.dataframe(anomalous_rows, use_container_width=True)
        else:
            st.success("✅ No anomalies detected for this engine.")
    else:
        st.info("Not enough cycles to compute anomaly scores (need at least window size).")


# ===========================================================================
# PAGE 4 — Model Metrics
# ===========================================================================
elif page == "📈 Model Metrics":
    st.title("📈 Model Evaluation Metrics")

    if metrics is None:
        st.warning(
            "No metrics found. Run the training pipeline:\n"
            "```bash\npython -m src.models.trainer --synthetic\n```"
        )
        st.stop()

    cfg = metrics.get("config", {})
    rul_m = metrics.get("rul", {})
    anom_m = metrics.get("anomaly", {})

    st.subheader("Training Configuration")
    st.json(cfg)

    st.markdown("---")
    st.subheader("🎯 RUL Prediction — Regression Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMSE ↓", rul_m.get("rmse", "N/A"), help="Root Mean Squared Error (lower is better)")
    col2.metric("MAE ↓", rul_m.get("mae", "N/A"), help="Mean Absolute Error")
    col3.metric("R² ↑", rul_m.get("r2", "N/A"), help="Coefficient of determination (higher is better)")
    col4.metric("NASA Score ↓", rul_m.get("nasa_score", "N/A"), help="Asymmetric NASA scoring function")

    st.markdown("---")
    st.subheader("🚨 Anomaly Detection — Classification Metrics")
    col1, col2 = st.columns(2)
    col1.metric("AUROC ↑", anom_m.get("auroc", "N/A"), help="Area Under ROC Curve (1.0 = perfect)")
    col2.metric("Avg Precision ↑", anom_m.get("avg_precision", "N/A"))

    st.markdown("---")
    st.info(
        "📄 **Methodology:** Piece-wise linear RUL targets clipped at 125 cycles (industry standard).  \n"
        "🏭 **Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation.  \n"
        "🔗 **Paper:** Saxena et al. (2008) — *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.*"
    )
