import sys
from pathlib import Path
from typing import Any, Union, Dict, List
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
    Make predictions using the trained demand forecasting model.
    """

    def __init__(self, model_path: Union[str, Path, None] = None) -> None:
        config = load_config()
        self.model_path = model_path or config["output"]["model_path"]

    def predict(
        self, input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
    ) -> pd.DataFrame:
        try:
            dataframe = self._to_dataframe(input_data)
            logging.info(f"Inference input received: {dataframe.shape}")

            # Feature processing pipeline
            cleaned_df = data_cleaner(dataframe)
            features = feature_engineering(cleaned_df)
            logging.info(f"Features prepared: {features.shape}")

            # Model application
            model = load_object(self.model_path)
            predictions = model.predict(features)

            # Return original data structured with the brand new predictions
            result = dataframe.copy()
            result["PredictedDemand"] = predictions
            return result

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _to_dataframe(
        input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        if isinstance(input_data, pd.DataFrame):
            return input_data.copy()
        if isinstance(input_data, (dict, list)):
            return pd.DataFrame(
                [input_data] if isinstance(input_data, dict) else input_data
            )
        raise TypeError("Unsupported data layout. Pass DataFrame, dict, or list.")
