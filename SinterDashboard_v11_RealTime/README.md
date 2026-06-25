# ⚙️ Sinter Quality Intelligence Platform

## Setup & Launch

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (one-time, ~2 minutes)
python model_training.py

# Launch dashboard
streamlit run app.py
```

## Files
- `app.py`                 — Main Streamlit dashboard (11 pages)
- `model_training.py`      — Model training pipeline (RF, XGB, LGB, CAT)
- `feature_engineering.py` — Domain-driven feature engineering
- `prediction_engine.py`   — Real-time prediction & BF suitability scoring
- `optimization_engine.py` — Optuna multi-objective optimization
- `artifacts/`             — Trained model artifacts (generated after training)
- `Outlier_Removed_Data.xlsx` — Clean training data

## Dashboard Pages
1. 🏠 Executive Dashboard    — KPIs, gauges, alerts
2. 📈 Quality Monitoring      — SPC charts with control limits
3. ⚙️ Process Monitoring      — Process parameter trends
4. 🔗 Correlation Analysis    — Interactive heatmap
5. 🧠 Feature Importance/SHAP — ML drivers & metallurgical interpretation
6. 🔍 Root Cause Analysis     — Contribution analysis & actions
7. 🎯 Quality Prediction      — Real-time slider-based prediction
8. 🔮 What-If Analysis        — Scenario comparison
9. ⚡ Optimization Engine     — Optuna Pareto optimization
10. 🔥 BF Suitability          — Blast furnace health panel
11. 💡 Management Insights     — Automated recommendations
