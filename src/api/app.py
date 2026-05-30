from typing import Any

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    BatchPredictionRequest,
    OptimizationInput,
    PredictionRequest,
    PredictionResponse,
)
from src.exception import CustomException
from src.pipeline.predict_pipeline import add_inventory_recommendations, predict_demand


app = FastAPI(title="Retail Demand Forecasting API", version="0.1.0")


def make_prediction_response(row: dict[str, Any]) -> PredictionResponse:
    return PredictionResponse(
        predicted_demand=float(row["PredictedDemand"]),
        daily_demand=to_optional_float(row.get("DailyDemand")),
        annual_demand=to_optional_float(row.get("AnnualDemand")),
        optimal_order_quantity=to_optional_float(row.get("OptimalOrderQuantity")),
        reorder_point=to_optional_float(row.get("ReorderPoint")),
        safety_stock=to_optional_float(row.get("SafetyStock")),
    )


def to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def predict_rows(rows: list[dict], optimization: OptimizationInput | None = None):
    result = predict_demand(rows)

    if optimization is not None:
        result = add_inventory_recommendations(
            predictions=result,
            ordering_cost=optimization.ordering_cost,
            holding_cost=optimization.holding_cost,
            lead_time_days=optimization.lead_time_days,
            safety_stock=optimization.safety_stock,
        )

    return result.to_dict(orient="records")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        rows = predict_rows(
            [request.data.to_model_input()],
            optimization=request.optimization,
        )
        return make_prediction_response(rows[0])

    except (CustomException, ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(request: BatchPredictionRequest) -> list[PredictionResponse]:
    try:
        input_rows = [row.to_model_input() for row in request.data]
        rows = predict_rows(input_rows, optimization=request.optimization)
        return [make_prediction_response(row) for row in rows]

    except (CustomException, ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
