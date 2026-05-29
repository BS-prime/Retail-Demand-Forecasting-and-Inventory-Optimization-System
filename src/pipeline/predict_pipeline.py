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


ROOT_DIR = Path(__file__).resolve().parents[2]


class PredictPipeline:
    """
    Load trained artifacts and generate demand predictions for raw retail rows.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        preprocessor_path: str | Path | None = None,
    ) -> None:
        config = load_config()
        self.model_path = model_path or config["output"]["model_path"]
        self.preprocessor_path = preprocessor_path or config["output"]["preprocessor_path"]

    def predict(self, input_data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]]):
        """
        Run the full inference flow on raw input data.
        """
        try:
            dataframe = self._to_dataframe(input_data)
            original_data = dataframe.copy()

            logging.info(f"Prediction input received: {dataframe.shape}")

            features = prepare_prediction_features(dataframe)

            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            transformed_features = preprocessor.transform(features)
            predictions = model.predict(transformed_features)

            result = original_data.copy()
            result["PredictedDemand"] = predictions

            logging.info(f"Prediction completed: {result.shape}")

            return result

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _to_dataframe(
        input_data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]]
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
        dataframe = dataframe.drop(columns=["Demand"], errors="ignore")

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
    parser = argparse.ArgumentParser(description="Run retail demand prediction pipeline.")
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
