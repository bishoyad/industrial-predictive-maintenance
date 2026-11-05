# How to Push to GitHub

## Step 1 — Create a new GitHub repository

Go to: https://github.com/new

- Repository name: `industrial-predictive-maintenance`
- Description: `End-to-end ML system for turbofan engine RUL prediction and anomaly detection (NASA C-MAPSS)`
- Public ✅
- Do NOT add README (we have one already)
- Click **Create repository**

## Step 2 — Push your code

Open your Terminal and run:

```bash
cd /Users/bishoyadel/.gemini/antigravity/scratch/industrial-predictive-maintenance

# Initialize git
git init
git add .
git commit -m "feat: initial project - predictive maintenance ML system"

# Connect to GitHub and push
git remote add origin https://github.com/bishoyad/industrial-predictive-maintenance.git
git branch -M main
git push -u origin main
```

## Step 3 — Run the CI pipeline

After pushing, go to:
  https://github.com/bishoyad/industrial-predictive-maintenance/actions

The CI workflow will automatically:
  - Install dependencies
  - Run all 15+ tests (test_preprocessor, test_models, test_api)
  - Report test coverage
  - Run linting

## Step 4 — Verify locally first

```bash
cd /Users/bishoyadel/.gemini/antigravity/scratch/industrial-predictive-maintenance

# Install dependencies
pip install -r requirements.txt

# Train on synthetic data (no downloads needed!)
python -m src.models.trainer --model xgboost --synthetic

# Run tests
pytest tests/ -v

# Start API
uvicorn api.main:app --reload --port 8000
# → Open: http://localhost:8000/docs

# Start Dashboard (new terminal tab)
streamlit run dashboard/app.py
# → Open: http://localhost:8501
```
