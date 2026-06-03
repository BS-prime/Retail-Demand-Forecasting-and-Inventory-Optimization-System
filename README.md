# Retail Demand Forecasting & Inventory Optimization System 🚀

A practical machine learning project for forecasting retail demand and translating those forecasts into inventory decisions. The system combines a training pipeline, a prediction pipeline, inventory optimization formulas, a FastAPI service, and a Streamlit dashboard.

## ✨ What This Project Does

- Forecasts product demand using historical retail data
- Trains and evaluates multiple regression models
- Supports inventory optimization with EOQ, reorder points, daily demand, and safety stock
- Serves predictions through a FastAPI backend
- Provides an interactive Streamlit UI for single and batch inference
- Generates SHAP explainability output for model interpretation

## 🧠 Core Features

- **End-to-end training pipeline**: ingestion, cleaning, feature engineering, preprocessing, model training, evaluation, and explainability
- **Reusable prediction pipeline**: loads saved model artifacts and returns demand predictions
- **Inventory planning layer**: converts forecasted demand into operational metrics
- **API-ready inference**: single-row prediction and batch CSV prediction endpoints
- **Dashboard workflow**: user-friendly controls for demand prediction and logistics parameters
- **Config-driven setup**: paths, model search space, and training settings live in `configs/config.yaml`

## 🏗️ Project Structure

```text
.
├── artifacts/                         # Saved preprocessing, metrics, features, and explainability outputs
├── configs/
│   └── config.yaml                     # Dataset, model, training, and output configuration
├── data/
│   ├── demand_forecasting.csv          # Source dataset
│   └── cleaned_demand_forecasting.parquet
├── models/
│   └── best_model.pkl                  # Saved best-performing model
├── notebooks/                          # EDA and model training notebooks
├── src/
│   ├── api/                            # FastAPI app and request/response schemas
│   ├── components/                     # Data, feature, training, evaluation, and SHAP components
│   ├── optimization_strategy/          # EOQ, reorder point, and safety stock logic
│   ├── pipeline/                       # Training and prediction orchestration
│   ├── services/                       # Inventory service layer
│   └── ui/                             # Streamlit app
├── pyproject.toml                      # Project metadata and dependencies
└── README.md
```

## ⚙️ Tech Stack

- Python 3.12+
- Pandas and NumPy
- scikit-learn
- XGBoost
- FastAPI
- Streamlit
- SHAP
- PyYAML
- Uvicorn

## 🚀 Quick Start

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install dependencies

```powershell
pip install -e .
```

If you use `uv`, you can also sync from the lockfile:

```powershell
uv sync
```

### 3. Train the model

```powershell
python src/pipeline/train_pipeline.py
```

The training pipeline will:

- Load data from `data/demand_forecasting.csv`
- Clean and engineer features
- Train candidate models from `configs/config.yaml`
- Evaluate models using regression metrics
- Save the best model to `models/best_model.pkl`
- Save preprocessing and metrics artifacts under `artifacts/`
- Generate a SHAP summary plot

## 🖥️ Run the Streamlit App

```powershell
streamlit run src/ui/main.py
```

Use the dashboard to:

- Enter a single product/store scenario
- Upload a CSV for batch inference
- Configure ordering cost, holding cost, supplier lead time, safety stock, and forecast horizon
- Export optimized inventory results

## 🌐 Run the FastAPI Service

```powershell
python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- API health check: `http://127.0.0.1:8000/health`
- Interactive API docs: `http://127.0.0.1:8000/docs`

## 🧪 API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Confirms the API is running |
| `POST` | `/predict` | Predicts demand and inventory metrics for one record |
| `POST` | `/predict/batch` | Accepts a CSV upload and returns optimized batch predictions |

### Example `/predict` Request

```json
{
  "data": {
    "Date": "2026-01-15T00:00:00",
    "StoreID": "S001",
    "ProductID": "P0001",
    "Category": "Electronics",
    "Region": "North",
    "Price": 99.99,
    "Discount": 10.0,
    "WeatherCondition": "Sunny",
    "Promotion": true,
    "CompetitorPricing": 95.0,
    "Seasonality": "Winter",
    "Epidemic": 0
  },
  "optimization": {
    "ordering_cost": 50.0,
    "holding_cost": 2.0,
    "lead_time_days": 7,
    "safety_stock": 10.0,
    "horizon_days": 365
  }
}
```

### Example `/predict` Response

```json
{
  "predicted_demand": 1240.58,
  "optimal_order_quantity": 249.06,
  "reorder_point": 33.79,
  "daily_demand": 3.4
}
```

## 📦 Inventory Optimization Logic

The system converts ML forecasts into inventory planning metrics:

- **Daily demand** = predicted demand divided by the selected forecast horizon
- **EOQ** = optimal order quantity based on annualized demand, ordering cost, and holding cost
- **Reorder point** = daily demand multiplied by supplier lead time, plus safety stock
- **Safety stock** = buffer inventory to reduce stockout risk

## 📊 Model Workflow

```text
Raw CSV Data
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
Preprocessing
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Saved Model + SHAP Explainability
    ↓
Prediction + Inventory Optimization
```

## 🔧 Configuration

Update `configs/config.yaml` to control:

- Input dataset path
- Candidate models and hyperparameter grids
- Train/test split
- Cross-validation settings
- Evaluation scoring
- Output paths for models, preprocessors, metrics, and explainability assets

## 📝 Notes

- Train the model before running fresh predictions if `models/best_model.pkl` is missing.
- Batch prediction CSV files should match the feature columns expected by the prediction pipeline.
- The Streamlit UI lives at `src/ui/main.py`.
- The FastAPI app lives at `src/api/app.py`.

## 🎯 Goal

Help retailers move from reactive inventory decisions to data-driven demand forecasting and smarter replenishment planning.
