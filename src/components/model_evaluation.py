# import modules
from math import inf
import sys
import json
from pathlib import Path

from src.config import load_config
from src.exception import CustomException
from src.logger import logging

# import libraries
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# locate root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# ========================================================================
# --- 1. Save metrics as json --
# ========================================================================

def save_evaluation_metrics(metrics_dict: dict) -> None:
    """
    Helper to save the evaluation metrics to a json file
    """

    try:
        config_file = load_config()
        file_path = ROOT_DIR / Path(config_file["output"]["metrics_path"])
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)

        logging.info(f"Metrics saved to filepath: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

# ========================================================================
# --- 2. Final Execution --
# ========================================================================

def model_evaluator(
    models: list,
    X_test,
    y_test: pd.Series,
)-> object:
    """
    Evaluate the models from the list and return the best model 
    and save the metrics of each model into a json file.
    """

    try:
        all_metrics: dict = {}
        best_model = None
        best_mae = float("inf")
        best_mse = float("inf")
        best_r2 = float("-inf")

        for model in models:
            model_name = type(model.named_steps["model"]).__name__ if hasattr(model, "named_steps") else type(model).__name__

            logging.info(f"Evaluation of model: {model_name} starting...")

            y_pred = model.predict(X_test)

            mae = round(mean_absolute_error(y_test, y_pred), 4)
            mse = round(mean_squared_error(y_test, y_pred), 4)
            r2 = round(r2_score(y_test, y_pred), 4)

            logging.info(f"Evaluation of {model_name} completed")

            metrics = {
                "MAE": float(mae),
                "MSE": float(mse),
                "r2_score": float(r2),
            }

            all_metrics[model_name] = metrics

            if r2 > best_r2 or (
                r2 == best_r2 and (mse < best_mse or (mse == best_mse and mae < best_mae))
            ):
                best_model = model
                best_mae = mae
                best_mse = mse
                best_r2 = r2

        save_evaluation_metrics(metrics_dict=all_metrics)

        return best_model

    except Exception as e:
        raise CustomException(e, sys)
