import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.components.data_cleaning import data_cleaner
from src.components.feature_engineering import feature_engineering
from src.config import load_config
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.optimization_strategy.optimization import InventoryOptimizer


ROOT_DIR = Path(__file__).resolve().parents[2]


class PredictPipeline:
    """
    Load trained artifacts and generate demand predictions for raw retail rows.
    Optionally incorporates inventory optimization strategies (EOQ, reorder points).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        ordering_cost: float | None = None,
        holding_cost: float | None = None,
        lead_time_days: int | None = None,
        safety_stock: float = 0,
        service_level: float = 0.95,
        enable_optimization: bool = False,
    ) -> None:

        config = load_config()
        self.model_path = (model_path or config["output"]["model_path"])

        # Optimization parameters
        self.ordering_cost = ordering_cost
        self.holding_cost = holding_cost
        self.lead_time_days = lead_time_days
        self.safety_stock = safety_stock
        self.service_level = service_level
        self.enable_optimization = enable_optimization

        if enable_optimization:
            if not all([ordering_cost, holding_cost, lead_time_days]):
                raise ValueError(
                    "ordering_cost, holding_cost, and lead_time_days are required when enable_optimization=True"
                )

    def predict(
        self,
        input_data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
        include_optimization: bool | None = None,
    ):

        try:
            dataframe = self._to_dataframe(input_data)
            original_data = dataframe.copy()

            logging.info(f"Prediction input received: {dataframe.shape}")

            features = prepare_prediction_features(dataframe)

            model = load_object(self.model_path)

            predictions = model.predict(features)

            result = original_data.copy()
            result["PredictedDemand"] = predictions

            # Apply optimization if enabled
            use_optimization = (
                include_optimization
                if include_optimization is not None
                else self.enable_optimization
            )
            if use_optimization:
                result = self._apply_inventory_optimization(result)

            logging.info(f"Prediction completed: {result.shape}")

            return result

        except Exception as e:
            raise CustomException(e, sys)

    def _apply_inventory_optimization(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        Apply inventory optimization calculations to prediction results.

        Adds columns for:
        - OptimalOrderQuantity (EOQ)
        - ReorderPoint
        - SafetyStock
        """
        try:
            optimizer = InventoryOptimizer()

            # Calculate EOQ for total predicted demand
            if self.ordering_cost and self.holding_cost:
                total_demand = result["PredictedDemand"].sum()
                eoq = optimizer.economic_order_quantity(
                    annual_demand=total_demand,
                    ordering_cost=self.ordering_cost,
                    holding_cost=self.holding_cost,
                )
                result["OptimalOrderQuantity"] = eoq

            # Calculate reorder point and safety stock
            if self.lead_time_days:
                # Calculate daily demand per row (useful for individual items)
                result["DailyDemand"] = (
                    result["PredictedDemand"] / 365
                )  # Convert annual to daily

                # Calculate reorder point
                result["ReorderPoint"] = result["DailyDemand"].apply(
                    lambda x: optimizer.reorder_point(
                        daily_demand=x,
                        lead_time_days=self.lead_time_days,
                        safety_stock=self.safety_stock,
                    )
                )

                # Add safety stock
                result["SafetyStock"] = self.safety_stock

            logging.info("Inventory optimization applied to predictions")

            return result

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _to_dataframe(
        input_data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Normalize supported input formats into a DataFrame.
        """
        try:
            if isinstance(input_data, pd.DataFrame):
                return input_data.copy()

            if isinstance(input_data, dict):
                return pd.DataFrame([input_data])

            if isinstance(input_data, list):
                return pd.DataFrame(input_data)

            raise TypeError(
                "input_data must be a pandas DataFrame, a row dictionary, or a list of row dictionaries"
            )

        except Exception as e:
            raise CustomException(e, sys)


def prepare_prediction_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Apply training-time cleaning and feature engineering before inference.
    """
    try:
        dataframe = data_cleaner(dataframe)
        dataframe = feature_engineering(dataframe)

        logging.info(f"Prediction features prepared: {dataframe.shape}")

        return dataframe

    except Exception as e:
        raise CustomException(e, sys)


def load_prediction_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load prediction rows from a CSV file.
    """
    try:
        path = Path(file_path)
        if not path.is_absolute():
            path = ROOT_DIR / path

        dataframe = pd.read_csv(path)

        logging.info(f"Prediction data loaded from: {path}")

        return dataframe

    except Exception as e:
        raise CustomException(e, sys)


def save_predictions(predictions: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save prediction results to a CSV file.
    """
    try:
        path = Path(output_path)
        if not path.is_absolute():
            path = ROOT_DIR / path

        path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(path, index=False)

        logging.info(f"Predictions saved to: {path}")

    except Exception as e:
        raise CustomException(e, sys)


def run_prediction_pipeline(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Execute inference from a CSV file and optionally save the predictions.
    """
    try:
        config = load_config()
        prediction_input_path = input_path or config["input"]["path"]

        dataframe = load_prediction_data(prediction_input_path)
        predictions = PredictPipeline().predict(dataframe)

        if output_path:
            save_predictions(predictions, output_path)

        return predictions

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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prediction_pipeline(input_path=args.input_path, output_path=args.output_path)
