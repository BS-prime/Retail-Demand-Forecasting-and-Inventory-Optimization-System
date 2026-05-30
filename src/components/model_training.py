# import modules
import sys

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException
from src.config import load_config

# import libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor


def model_trainer(X_train, y_train, preprocessor):
    """
    Train multiple models and return in a list.
    """

    try:
        config_file = load_config()

        algo_map = {
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "ElasticNet": ElasticNet,
            "XGBRegressor": XGBRegressor,
        }

        models: list = []

        for model_config in config_file["models"].values():
            model_name = model_config["type"]
            model = Pipeline(
                [
                    ("preprocessor", clone(preprocessor)),
                    ("model", algo_map[model_name]()),
                ]
            )
            param_grid = {
                f"model__{parameter}": values
                for parameter, values in model_config["params"].items()
            }

            logging.info(f"Training model: {model_name}")
            
            # implementing time series split for cross validation
            time_series_split = TimeSeriesSplit(n_splits=config_file["training"]["cv"], gap=1)

            grid = GridSearchCV(
                model,
                param_grid=param_grid,
                cv=time_series_split,
                verbose=config_file["training"]["verbose"],
                scoring=config_file["training"]["scoring"],
            )

            grid.fit(X_train, y_train)

            logging.info(f"{model_name} trained successfully")

            best_model = grid.best_estimator_

            models.append(best_model)

            logging.info(f"{model_name} pipeline appended to the list successfully.")

        return models

    except Exception as e:
        raise CustomException(e, sys)
