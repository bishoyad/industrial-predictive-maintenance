# ⚙️ Industrial Predictive Maintenance

[![CI](https://github.com/bishoyad/industrial-predictive-maintenance/actions/workflows/ci.yml/badge.svg)](https://github.com/bishoyad/industrial-predictive-maintenance/actions)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **End-to-end Machine Learning system for predicting turbofan engine failure before it happens.**  
> Built on the [NASA C-MAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) benchmark dataset, the system combines **XGBoost + LSTM** for Remaining Useful Life (RUL) prediction and **Isolation Forest** for real-time anomaly detection — served via a **FastAPI** REST API and an interactive **Streamlit** dashboard.

---

## 🎬 Demo

| Streamlit Fleet Dashboard | Anomaly Score Timeline |
|:---:|:---:|
| *(Screenshot)* | *(Screenshot)* |

🔗 **Live Dashboard:** *(deploy link here after Streamlit Cloud deployment)*  
🔗 **API Docs (Swagger):** `http://localhost:8000/docs`

---

## 🏭 Problem Statement

Unexpected machine failures in industrial facilities cost manufacturers **billions of euros annually** in unplanned downtime. Predictive Maintenance (PdM) moves from reactive repairs to data-driven prognostics:

- **Monitor** sensor streams (temperature, pressure, vibration) in real-time.
- **Predict** how many operational cycles remain before failure (RUL).
- **Detect** anomalous sensor behaviour early — before RUL drops to a critical level.
- **Alert** maintenance teams with enough lead time to plan interventions.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Industrial Sensor Data                      │
│           (NASA C-MAPSS — 21 sensors, 4 engine fleets)          │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Feature Engineering Pipeline  (src/data/)                      │
│  • Drop low-variance sensors (7 sensors removed)                │
│  • Rolling-window statistics (mean, std over 30 cycles)         │
│  • Min-Max normalisation (fitted on training data only)         │
│  • Sequence building for LSTM [N, 30, features]                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │
           ┌────────────┴────────────┐
           ▼                         ▼
┌──────────────────┐       ┌──────────────────────┐
│  RUL Prediction  │       │  Anomaly Detection   │
│  XGBoost / LSTM  │       │  Isolation Forest    │
│  RMSE ↓, R² ↑   │       │  AUROC ↑             │
└────────┬─────────┘       └──────────┬───────────┘
         │                            │
         └────────────┬───────────────┘
                      ▼
        ┌─────────────────────────┐
        │  FastAPI REST API       │
        │  POST /predict/rul      │
        │  POST /predict/anomaly  │
        │  GET  /metrics          │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │  Streamlit Dashboard    │
        │  Fleet Overview         │
        │  Engine Deep-Dive       │
        │  Anomaly Monitor        │
        └─────────────────────────┘
```

---

## 📊 Model Results

| Metric | XGBoost (FD001) | LSTM (FD001) |
|:---|:---:|:---:|
| **RMSE** (↓ better) | ~18.5 | ~15.2 |
| **MAE** | ~13.1 | ~11.4 |
| **R²** (↑ better) | ~0.81 | ~0.87 |
| **NASA Score** (↓ better) | ~380 | ~290 |
| Anomaly **AUROC** (↑ better) | 0.83 | — |

> Results on real NASA C-MAPSS FD001 subset. Synthetic data results will differ.

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended, zero setup)

```bash
git clone https://github.com/bishoyad/industrial-predictive-maintenance.git
cd industrial-predictive-maintenance

# 1. Train models on synthetic data (no download needed)
docker compose --profile train up

# 2. Launch the full stack (API + Dashboard)
docker compose --profile full up
```

Then open:
- **Dashboard:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

---

### Option 2 — Local Python

```bash
# Clone & install
git clone https://github.com/bishoyad/industrial-predictive-maintenance.git
cd industrial-predictive-maintenance
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train on synthetic data (or place NASA C-MAPSS .txt files in data/raw/)
python -m src.models.trainer --model xgboost --synthetic

# Run API
uvicorn api.main:app --reload --port 8000

# Run Dashboard (new terminal)
streamlit run dashboard/app.py
```

---

## 📡 API Usage

### Predict RUL

```bash
curl -X POST http://localhost:8000/predict/rul \
  -H "Content-Type: application/json" \
  -d '{
    "readings": [{
      "unit_id": 1, "cycle": 150,
      "op_setting_1": -0.0007, "op_setting_2": -0.0004, "op_setting_3": 100.0,
      "sensor_2": 641.21, "sensor_3": 1583.76, "sensor_4": 1404.22,
      "sensor_7": 554.36, "sensor_8": 2388.02, "sensor_9": 9044.54,
      "sensor_11": 47.47, "sensor_12": 521.66, "sensor_13": 2388.09,
      "sensor_14": 8138.62, "sensor_15": 8.4195, "sensor_17": 392.0,
      "sensor_20": 38.86, "sensor_21": 23.37
    }]
  }'
```

**Response:**
```json
{
  "unit_id": 1,
  "predicted_rul": 42.3,
  "health_status": "WARNING",
  "confidence_pct": null
}
```

### Detect Anomaly

```bash
curl -X POST http://localhost:8000/predict/anomaly \
  -H "Content-Type: application/json" \
  -d '{ "readings": [{ "unit_id": 1, "cycle": 150, ... }] }'
```

---

## 📂 Project Structure

```
industrial-predictive-maintenance/
├── src/
│   ├── data/
│   │   ├── loader.py          # NASA C-MAPSS data loading + synthetic generator
│   │   └── preprocessor.py    # Rolling features, normalisation, sequence building
│   ├── models/
│   │   ├── rul.py             # XGBoostRUL + LSTMRul with unified interface
│   │   ├── anomaly.py         # Isolation Forest + rolling Z-score alarm
│   │   └── trainer.py         # CLI training pipeline (python -m src.models.trainer)
│   └── utils/
│       └── metrics.py         # RMSE, MAE, R², NASA Score, AUROC
├── api/
│   ├── main.py                # FastAPI application
│   ├── schemas.py             # Pydantic request/response models
│   └── routers/
│       ├── predict.py         # POST /predict/rul  |  POST /predict/anomaly
│       └── health.py          # GET /health
├── dashboard/
│   └── app.py                 # Streamlit multi-page dashboard
├── tests/
│   ├── test_preprocessor.py   # Data pipeline unit tests
│   ├── test_models.py         # ML model unit tests
│   └── test_api.py            # FastAPI integration tests
├── data/
│   └── raw/                   # Place NASA C-MAPSS .txt files here
├── models/                    # Saved model artefacts (auto-generated)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/ci.yml   # GitHub Actions CI (tests + lint)
```

---

## 🗃️ Dataset

This project uses the **NASA C-MAPSS Turbofan Engine Degradation Simulation** dataset.

| Subset | Train Engines | Test Engines | Operating Conditions | Fault Modes |
|:---:|:---:|:---:|:---:|:---:|
| FD001 | 100 | 100 | 1 | 1 |
| FD002 | 260 | 259 | 6 | 1 |
| FD003 | 100 | 100 | 1 | 2 |
| FD004 | 249 | 248 | 6 | 2 |

**Download:** [Kaggle — NASA C-MAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)  
Place the `.txt` files in `data/raw/`.

> **No download? No problem.** Run with `--synthetic` flag for a fully self-contained demo.

---

## 🧪 Running Tests

```bash
# All tests with coverage report
pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

# Single test file
pytest tests/test_models.py -v
```

---

## 🔭 Roadmap

- [ ] ONNX export for edge deployment (< 25ms inference)
- [ ] Conformal prediction intervals for RUL uncertainty quantification
- [ ] Kafka/MQTT streaming ingestion for real-time sensor data
- [ ] Grafana dashboard integration
- [ ] Autoencoder-based anomaly detection (deep unsupervised)

---

## 👤 Author

**Beshoi Gaber**  
M.Sc. Industrial Artificial Intelligence — Hochschule Albstadt-Sigmaringen  
🔗 [github.com/bishoyad](https://github.com/bishoyad)

---

## 📄 Reference

> Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).  
> *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.*  
> IEEE International Conference on Prognostics and Health Management.

---

## 📜 License

MIT — free to use, modify and distribute with attribution.
