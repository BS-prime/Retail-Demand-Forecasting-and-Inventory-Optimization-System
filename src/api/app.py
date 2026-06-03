from contextlib import asynccontextmanager
import io
from typing import Any, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import pandas as pd

from src.api.schemas import PredictionRequest, PredictionResponse
from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline
from src.services.inventory_service import InventoryService


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    # Instantiate and warm up the pure ML model artifact on application boot
    fastapi_app.state.predictor = PredictPipeline()
    yield


app = FastAPI(
    title="Retail Demand Forecasting API",
    description="API serving production ML predictions and operations-research inventory optimizations.",
    version="0.2.0",
    lifespan=lifespan,
)


def run_optimization_service(
    prediction_df: pd.DataFrame, opt_config: dict, horizon_days: int
) -> List[dict]:
    """
    Helper to make inventory service calls more concise within the endpoint logic, while ensuring that the time horizon is safely handled across the board.
    """
    inventory_service = InventoryService(
        ordering_cost=opt_config["ordering_cost"],
        holding_cost=opt_config["holding_cost"],
        lead_time_days=opt_config["lead_time_days"],
        safety_stock=opt_config.get("safety_stock", 0.0),
    )

    # Process through the service layer (which safely standardizes the horizons!)
    optimized_df = inventory_service.process_operational_plan(
        prediction_df=prediction_df, days_in_horizon=horizon_days
    )
    return optimized_df.to_dict(orient="records")


def make_prediction_response(row: dict[str, Any]) -> PredictionResponse:
    return PredictionResponse(
        predicted_demand=float(row["PredictedDemand"]),
        optimal_order_quantity=float(row["OptimalOrderQuantity"]),
        reorder_point=float(row["ReorderPoint"]),
        daily_demand=float(row["DailyDemand"]),
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "Retail Demand Forecasting API is up and running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        # 1. Pure ML inference execution
        input_data = [request.data.to_model_input()]
        raw_predictions = app.state.predictor.predict(input_data)

        # 2. Extract optimization and hand off to the operational service layer
        opt_dict = request.optimization.model_dump()
        horizon_days = opt_dict.pop("horizon_days")

        optimized_records = run_optimization_service(
            raw_predictions, opt_dict, horizon_days
        )
        return make_prediction_response(optimized_records[0])

    except (CustomException, ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(
    upload_file: UploadFile = File(...),
    ordering_cost: float = Form(...),
    holding_cost: float = Form(...),
    lead_time_days: int = Form(...),
    safety_stock: float = Form(0.0),
    horizon_days: int = Form(365),
) -> List[PredictionResponse]:
    """Inbiound CSV parsing combined with form parameters for real-time inventory translation."""
    try:
        # Read the raw uploaded CSV file into a pandas dataframe
        contents = await upload_file.read()
        df = pd.read_csv(io.BytesIO(contents))

        if df.empty:
            raise ValueError("The provided CSV data file is empty.")

        # 1. Generate core predictions using our startup pipeline instance
        raw_predictions = app.state.predictor.predict(df)

        # 2. Consolidate functional arguments for the service layout
        opt_config = {
            "ordering_cost": ordering_cost,
            "holding_cost": holding_cost,
            "lead_time_days": lead_time_days,
            "safety_stock": safety_stock,
        }

        # 3. Route through optimization service
        optimized_records = run_optimization_service(
            raw_predictions, opt_config, horizon_days
        )
        return [make_prediction_response(row) for row in optimized_records]

    except (CustomException, ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
