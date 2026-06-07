# Retail Demand Forecasting and Inventory Optimization System

A full-stack machine learning project that forecasts retail product demand and converts those forecasts into practical inventory planning recommendations. The system combines a reproducible training pipeline, time-aware model validation, SHAP explainability, Economic Order Quantity (EOQ) optimization, reorder point planning, a FastAPI inference service, and a Streamlit dashboard for interactive business use.

## Project Overview

Retail teams need more than a demand prediction. They need to know how much inventory to order, when to reorder, and how supplier lead time or safety stock decisions affect replenishment. This project addresses that workflow end to end:

- Train demand forecasting models on historical retail data.
- Select the best-performing model using regression metrics.
- Generate explainability artifacts for model interpretation.
- Serve single-record and batch demand predictions through an API.
- Translate predicted demand into inventory metrics such as EOQ, daily demand, reorder point, and safety stock.
- Provide a Streamlit interface for business-facing prediction and optimization workflows.

## Key Highlights

- **End-to-end ML pipeline**: ingestion, cleaning, feature engineering, preprocessing, model training, model evaluation, artifact saving, and explainability.
- **Strong model benchmark**: XGBoost achieved the best recorded performance with `R2 = 0.9671`, `MAE = 5.5242`, and `MSE = 63.9249`.
- **Time-aware validation**: the training process uses chronological splitting and `TimeSeriesSplit` cross-validation to better respect the temporal structure of demand data.
- **Leakage-conscious feature engineering**: prediction-time leakage columns such as `Units Sold`, `Units Ordered`, and `Inventory Level` are dropped before modeling.
- **Business-ready optimization layer**: demand forecasts are converted into EOQ, reorder point, daily run rate, annualized demand, and safety stock outputs.
- **FastAPI inference service**: exposes health, single prediction, and batch CSV prediction endpoints with Pydantic validation.
- **Streamlit decision dashboard**: supports single scenario analysis, batch file processing, configurable logistics assumptions, and CSV export.
- **Explainable ML**: saves a SHAP summary plot to identify the features that influence model predictions.
- **Config-driven design**: model grids, training settings, input paths, and artifact paths are managed through `configs/config.yaml`.

## Dataset

The project uses a retail demand forecasting dataset with **76,000 records**. Each record includes product, store, pricing, promotional, seasonal, weather, regional, and disruption-related features.

Dataset columns include:

| Column | Description |
| --- | --- |
| `Date` | Transaction or demand observation date |
| `Store ID` | Store identifier |
| `Product ID` | Product identifier |
| `Category` | Product category |
| `Region` | Sales region |
| `Price` | Retail price |
| `Discount` | Discount percentage |
| `Weather Condition` | Weather context |
| `Promotion` | Promotion indicator |
| `Competitor Pricing` | Competitor price signal |
| `Seasonality` | Seasonal context |
| `Epidemic` | Disruption indicator |
| `Demand` | Target variable |

## Model Performance

The training pipeline evaluates multiple regression models and saves the metrics to `artifacts/metrics/metrics.json`.

| Model | MAE | MSE | R2 Score |
| --- | ---: | ---: | ---: |
| Linear Regression | 25.6901 | 1094.6899 | 0.4368 |
| Ridge | 25.6885 | 1094.7736 | 0.4367 |
| Lasso | 25.6923 | 1095.0678 | 0.4366 |
| ElasticNet | 25.6926 | 1096.2320 | 0.4360 |
| XGBRegressor | **5.5242** | **63.9249** | **0.9671** |

The best model is saved to `models/best_model.pkl` and reused by the prediction pipeline, API, and Streamlit application.

## System Architecture

```text
Raw Retail CSV
    |
    v
Data Ingestion
    |
    v
Data Cleaning
    |
    v
Feature Engineering
    |-- Date features: year, month, day, weekday
    |-- Discounted price
    |-- Leakage column removal
    v
Preprocessing
    |-- Numeric imputation and scaling
    |-- Categorical imputation and one-hot encoding
    v
Model Training
    |-- LinearRegression
    |-- Ridge
    |-- Lasso
    |-- ElasticNet
    |-- XGBRegressor
    v
Model Evaluation and Artifact Saving
    |-- best_model.pkl
    |-- preprocessor.pkl
    |-- metrics.json
    |-- shap_summary.png
    v
Prediction Pipeline
    |
    v
Inventory Optimization
    |-- Daily demand
    |-- Annualized demand
    |-- EOQ
    |-- Reorder point
    |-- Safety stock
    v
FastAPI Service / Streamlit Dashboard
```

## Inventory Optimization Logic

The project connects machine learning output with inventory planning formulas:

| Metric | Purpose |
| --- | --- |
| Daily Demand | Converts predicted demand into an operational daily run rate |
| Annualized Demand | Standardizes demand for EOQ calculations |
| EOQ | Calculates the cost-efficient order quantity using demand, ordering cost, and holding cost |
| Reorder Point | Identifies the inventory threshold for placing the next order |
| Safety Stock | Adds a buffer against demand or supply uncertainty |

Core formulas:

```text
Daily Demand = Predicted Demand / Forecast Horizon
Annualized Demand = Daily Demand * 365
EOQ = sqrt((2 * Annualized Demand * Ordering Cost) / Holding Cost)
Reorder Point = (Daily Demand * Lead Time Days) + Safety Stock
```

## Application Interfaces

### Streamlit Dashboard

The Streamlit app provides an interactive decision interface for demand forecasting and replenishment planning.

Capabilities:

- Single product/store scenario prediction.
- Batch CSV prediction.
- Configurable ordering cost, holding cost, supplier lead time, safety stock, and forecast horizon.
- Operational metrics displayed as dashboard KPIs.
- Exportable optimized inventory plan.

Run it with:

```powershell
streamlit run src/ui/main.py
```

### FastAPI Service

The API exposes model inference and inventory optimization as structured HTTP endpoints.

Run it with:

```powershell
python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Available endpoints:

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Confirms the API is running |
| `POST` | `/predict` | Predicts demand and inventory metrics for one record |
| `POST` | `/predict/batch` | Accepts a CSV file and returns optimized predictions for multiple records |

Once the server is running:

- API health check: `http://127.0.0.1:8000/health`
- Interactive documentation: `http://127.0.0.1:8000/docs`

Example `/predict` request:

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

Example response:

```json
{
  "predicted_demand": 1240.58,
  "optimal_order_quantity": 249.06,
  "reorder_point": 33.79,
  "daily_demand": 3.4
}
```

## Project Structure

```text
.
|-- artifacts/
|   |-- explainability/
|   |   `-- shap_summary.png
|   |-- features/
|   |   |-- train.csv
|   |   `-- test.csv
|   |-- metrics/
|   |   `-- metrics.json
|   `-- preprocessor/
|       `-- preprocessor.pkl
|-- configs/
|   `-- config.yaml
|-- data/
|   |-- demand_forecasting.csv
|   `-- cleaned_demand_forecasting.parquet
|-- models/
|   `-- best_model.pkl
|-- notebooks/
|   |-- eda.ipynb
|   `-- model_training_and_evaluation.ipynb
|-- src/
|   |-- api/
|   |   |-- app.py
|   |   `-- schemas.py
|   |-- components/
|   |   |-- data_cleaning.py
|   |   |-- data_ingestion.py
|   |   |-- data_preprocessing.py
|   |   |-- feature_engineering.py
|   |   |-- model_evaluation.py
|   |   |-- model_explainability.py
|   |   `-- model_training.py
|   |-- optimization_strategy/
|   |   `-- optimization.py
|   |-- pipeline/
|   |   |-- predict_pipeline.py
|   |   `-- train_pipeline.py
|   |-- services/
|   |   `-- inventory_service.py
|   `-- ui/
|       `-- main.py
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

## Technology Stack

| Area | Tools |
| --- | --- |
| Language | Python 3.12+ |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost |
| Model selection | GridSearchCV, TimeSeriesSplit |
| Preprocessing | ColumnTransformer, SimpleImputer, StandardScaler, OneHotEncoder |
| Explainability | SHAP, Matplotlib |
| API | FastAPI, Pydantic, Uvicorn |
| UI | Streamlit |
| Configuration | PyYAML |
| Packaging | pyproject.toml, uv |

## Getting Started

### 1. Clone the repository

```powershell
git clone <repository-url>
cd Retail-Demand-Forecasting-and-Inventory-Optimization-System
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install dependencies

```powershell
pip install -e .
```

If you use `uv`, you can install from the lockfile:

```powershell
uv sync
```

### 4. Train the model

```powershell
python src/pipeline/train_pipeline.py
```

Training will generate or update:

- `models/best_model.pkl`
- `artifacts/preprocessor/preprocessor.pkl`
- `artifacts/metrics/metrics.json`
- `artifacts/features/train.csv`
- `artifacts/features/test.csv`
- `artifacts/explainability/shap_summary.png`

### 5. Run inference

Use the Streamlit dashboard:

```powershell
streamlit run src/ui/main.py
```

Or start the FastAPI service:

```powershell
python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

## Configuration

The central configuration file is `configs/config.yaml`.

It controls:

- Input dataset path.
- Model families and hyperparameter grids.
- Train/test split settings.
- Cross-validation settings.
- Scoring metric.
- Output paths for models, preprocessors, metrics, transformed features, and explainability assets.

## Important Entry Points

| Task | File |
| --- | --- |
| Train models | `src/pipeline/train_pipeline.py` |
| Run reusable inference | `src/pipeline/predict_pipeline.py` |
| Start FastAPI app | `src/api/app.py` |
| Validate API schemas | `src/api/schemas.py` |
| Run Streamlit app | `src/ui/main.py` |
| Inventory formulas | `src/optimization_strategy/optimization.py` |
| Forecast-to-plan service | `src/services/inventory_service.py` |
| Update training configuration | `configs/config.yaml` |

## Notes

- Train the model before running new predictions if `models/best_model.pkl` is missing.
- Batch prediction files should match the feature format expected by the prediction pipeline.
- The Streamlit app entrypoint is `src/ui/main.py`.
- The FastAPI app entrypoint is `src/api/app.py`.
- The repository root `main.py` is not the primary application entrypoint.

## Why This Project Matters

This project demonstrates how demand forecasting can be operationalized beyond model training. It turns historical retail data into forecasts, converts those forecasts into inventory decisions, and exposes the workflow through both an API and an interactive dashboard. The result is a practical analytics system that connects machine learning, explainability, and supply chain decision support.
