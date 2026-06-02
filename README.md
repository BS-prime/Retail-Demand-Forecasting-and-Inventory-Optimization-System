# Retail Demand Forecasting and Inventory Optimization System 🚀

A Python-based retail analytics system for demand forecasting, inventory optimization, model explainability, and easy inference through both a Streamlit UI and a FastAPI service.

## 🌟 What this project does

- Predicts retail demand from historical sales and store data
- Trains regression models using scikit-learn and XGBoost
- Supports inventory optimization using Economic Order Quantity (EOQ), reorder points, and safety stock
- Provides a Streamlit dashboard for single-row and batch inference
- Exposes a FastAPI service for structured prediction and batch prediction
- Generates explainability output with SHAP summaries

## 🧩 Key features

- `src/pipeline/train_pipeline.py` — trains the model with data ingestion, cleaning, feature engineering, evaluation, and explainability
- `src/pipeline/predict_pipeline.py` — loads trained artifacts and runs predictions
- `src/optimization_strategy/optimization.py` — calculates EOQ, reorder points, and safety stock
- `src/api/app.py` — FastAPI endpoints for single and batch prediction
- `src/ui/main.py` — Streamlit app for interactive demand prediction and inventory strategy
- `configs/config.yaml` — configuration for input data, models, training, and artifact paths

## 🚀 Quick start

### 1. Create your Python environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

> If you don't want the editable install, you can also install the dependencies directly with `pip install -r requirements.txt` after generating one from `pyproject.toml`.

### 2. Train the model

```bash
python src/pipeline/train_pipeline.py
```

This pipeline will:
- ingest `data/demand_forecasting.csv`
- clean and engineer features
- train candidate models
- evaluate the best model
- save the model and preprocessor to `artifacts/` and `models/`
- create a SHAP explainability plot

### 3. Run the Streamlit UI

```bash
streamlit run src/ui/main.py
```

Then open the browser page Streamlit provides to:
- submit a single prediction
- upload a CSV for batch prediction
- configure mandatory inventory optimization parameters

### 4. Run the FastAPI service

```bash
python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Then visit:
- `http://127.0.0.1:8000/health` for status
- `http://127.0.0.1:8000/docs` for API documentation

## 🧪 API endpoints

- `GET /health` — health check
- `POST /predict` — single row prediction
- `POST /predict/batch` — batch CSV prediction

### Example request body for `/predict`

```json
{
  "data": {
    "Date": "2025-12-31T00:00:00",
    "StoreID": "S001",
    "ProductID": "P0001",
    "Category": "Electronics",
    "Region": "North",
    "InventoryLevel": 100,
    "UnitsSold": 10,
    "UnitsOrdered": 5,
    "Price": 99.99,
    "Discount": 10.0,
    "WeatherCondition": "Sunny",
    "Promotion": false,
    "CompetitorPricing": 95.0,
    "Seasonality": "Winter",
    "Epidemic": 0
  }
}
```

### Example optimization block

```json
{
  "optimization": {
    "ordering_cost": 50.0,
    "holding_cost": 2.0,
    "lead_time_days": 7,
    "safety_stock": 10.0
  }
}
```

## 📁 Project structure

- `src/`
  - `api/` — FastAPI service and request/response schemas
  - `components/` — data ingestion, cleaning, preprocessing, feature engineering, training, evaluation, explainability
  - `pipeline/` — training and prediction pipeline orchestration
  - `optimization_strategy/` — inventory optimization calculations
  - `ui/` — Streamlit interface
- `configs/` — YAML configuration settings
- `data/` — raw demand forecasting dataset
- `models/` — saved model artifacts
- `artifacts/` — saved features, metrics, explainability, and preprocessor files
- `notebooks/` — exploratory data analysis and model evaluation notebooks

## 🛠️ Dependencies

Managed through `pyproject.toml`.

Primary packages include:
- `fastapi`
- `streamlit`
- `scikit-learn`
- `xgboost`
- `pandas`
- `numpy`
- `pyyaml`
- `shap`
- `uvicorn`

## 💡 Notes

- The main app file at the repo root is empty; use `src/ui/main.py` for Streamlit and `src/api/app.py` for the API.
- Configuration is controlled by `configs/config.yaml`; update paths, model search space, and training settings there.
- For better performance, ensure the model artifacts are generated before running predictions.

## 🎯 Goal

Build a practical retail forecasting system that combines demand prediction with inventory optimization and interactive inference.
