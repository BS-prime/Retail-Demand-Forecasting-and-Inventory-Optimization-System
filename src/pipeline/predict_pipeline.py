import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.components.data_cleaning import data_cleaner
from src.components.data_preprocessing import LEAKAGE_COLUMNS
from src.components.feature_engineering import feature_engineering
from src.config import load_config
from src.exception import CustomException
from src.logger import logging
from src.optimization_strategy.optimization import InventoryOptimizer
from src.utils import load_object


ROOT_DIR = Path(__file__).resolve().parents[2]


def to_dataframe(
    input_data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()

    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])

    if isinstance(input_data, list):
        return pd.DataFrame(input_data)

    raise TypeError(
        "input_data must be a pandas DataFrame, a row dictionary, or a list of row dictionaries"
    )


def prepare_prediction_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = data_cleaner(dataframe)
    dataframe = feature_engineering(dataframe)
    dataframe = dataframe.drop(columns=["Demand", *LEAKAGE_COLUMNS], errors="ignore")

    logging.info(f"Prediction features prepared: {dataframe.shape}")
    return dataframe


def predict_demand(
    input_data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    model_path: str | Path | None = None,
    preprocessor_path: str | Path | None = None,
) -> pd.DataFrame:
    config = load_config()
    model_path = model_path or config["output"]["model_path"]
    preprocessor_path = preprocessor_path or config["output"]["preprocessor_path"]

    dataframe = to_dataframe(input_data)
    original_data = dataframe.copy()
    features = prepare_prediction_features(dataframe)

    model = load_object(model_path)

    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        predictions = model.predict(features)
    else:
        preprocessor = load_object(preprocessor_path)
        transformed_features = preprocessor.transform(features)
        predictions = model.predict(transformed_features)

    result = original_data.copy()
    result["PredictedDemand"] = predictions

    logging.info(f"Prediction completed: {result.shape}")
    return result


def add_inventory_recommendations(
    predictions: pd.DataFrame,
    ordering_cost: float,
    holding_cost: float,
    lead_time_days: float,
    safety_stock: float = 0,
    days_per_year: int = 365,
) -> pd.DataFrame:
    optimizer = InventoryOptimizer()
    result = predictions.copy()

    result["PredictedDemand"] = result["PredictedDemand"].clip(lower=0)
    result["DailyDemand"] = result["PredictedDemand"]
    result["AnnualDemand"] = result["DailyDemand"] * days_per_year
    result["SafetyStock"] = safety_stock
    result["OptimalOrderQuantity"] = result["AnnualDemand"].apply(
        lambda annual_demand: (
            optimizer.economic_order_quantity(
                annual_demand=annual_demand,
                ordering_cost=ordering_cost,
                holding_cost=holding_cost,
            )
            if annual_demand > 0
            else 0
        )
    )
    result["ReorderPoint"] = result["DailyDemand"].apply(
        lambda daily_demand: optimizer.reorder_point(
            daily_demand=daily_demand,
            lead_time_days=lead_time_days,
            safety_stock=safety_stock,
        )
    )

    logging.info(f"Inventory recommendations added: {result.shape}")
    return result


def load_prediction_data(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.is_absolute():
        path = ROOT_DIR / path

    dataframe = pd.read_csv(path)
    logging.info(f"Prediction data loaded from: {path}")
    return dataframe


def save_predictions(predictions: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    if not path.is_absolute():
        path = ROOT_DIR / path

    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    logging.info(f"Predictions saved to: {path}")


def run_prediction_pipeline(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    optimize: bool = False,
    ordering_cost: float | None = None,
    holding_cost: float | None = None,
    lead_time_days: float | None = None,
    safety_stock: float = 0,
) -> pd.DataFrame:
    try:
        config = load_config()
        prediction_input_path = input_path or config["input"]["path"]

        dataframe = load_prediction_data(prediction_input_path)
        predictions = predict_demand(dataframe)

        if optimize:
            if ordering_cost is None or holding_cost is None or lead_time_days is None:
                raise ValueError(
                    "ordering_cost, holding_cost, and lead_time_days are required when optimize=True"
                )

            predictions = add_inventory_recommendations(
                predictions=predictions,
                ordering_cost=ordering_cost,
                holding_cost=holding_cost,
                lead_time_days=lead_time_days,
                safety_stock=safety_stock,
            )

        if output_path:
            save_predictions(predictions, output_path)

        return predictions

    except Exception as e:
        raise CustomException(e, sys)


class PredictPipeline:
    """
    Thin wrapper kept for API compatibility. The script logic lives in functions above.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        preprocessor_path: str | Path | None = None,
        ordering_cost: float | None = None,
        holding_cost: float | None = None,
        lead_time_days: float | None = None,
        safety_stock: float = 0,
        enable_optimization: bool = False,
    ) -> None:
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.ordering_cost = ordering_cost
        self.holding_cost = holding_cost
        self.lead_time_days = lead_time_days
        self.safety_stock = safety_stock
        self.enable_optimization = enable_optimization

    def predict(
        self,
        input_data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
        include_optimization: bool | None = None,
    ) -> pd.DataFrame:
        try:
            result = predict_demand(
                input_data=input_data,
                model_path=self.model_path,
                preprocessor_path=self.preprocessor_path,
            )
            optimize = (
                self.enable_optimization
                if include_optimization is None
                else include_optimization
            )

            if not optimize:
                return result

            if (
                self.ordering_cost is None
                or self.holding_cost is None
                or self.lead_time_days is None
            ):
                raise ValueError(
                    "ordering_cost, holding_cost, and lead_time_days are required for optimization"
                )

            return add_inventory_recommendations(
                predictions=result,
                ordering_cost=self.ordering_cost,
                holding_cost=self.holding_cost,
                lead_time_days=self.lead_time_days,
                safety_stock=self.safety_stock,
            )

        except Exception as e:
            raise CustomException(e, sys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retail demand prediction pipeline."
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        default=None,
        help="CSV file containing raw rows for prediction. Defaults to config input path.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default="artifacts/predictions/predictions.csv",
        help="CSV path where predictions will be saved.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Include EOQ and reorder point recommendations.",
    )
    parser.add_argument("--ordering-cost", type=float, default=None)
    parser.add_argument("--holding-cost", type=float, default=None)
    parser.add_argument("--lead-time-days", type=float, default=None)
    parser.add_argument("--safety-stock", type=float, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prediction_pipeline(
        input_path=args.input_path,
        output_path=args.output_path,
        optimize=args.optimize,
        ordering_cost=args.ordering_cost,
        holding_cost=args.holding_cost,
        lead_time_days=args.lead_time_days,
        safety_stock=args.safety_stock,
    )
