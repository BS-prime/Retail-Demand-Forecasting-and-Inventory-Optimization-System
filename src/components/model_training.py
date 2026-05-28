# import modules
import sys

from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomException
from src.config import load_config

# import libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor


def model_trainer(preprocessed_X_train, y_train):
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
            model = algo_map[model_name](
                random_state=config_file["training"]["random_state"]
            )

            logging.info(f"Training model: {model_name}")

            grid = GridSearchCV(
                model,
                param_grid=model_config["params"],
                cv=config_file["training"]["cv"],
                verbose=config_file["training"]["verbose"],
                scoring=config_file["training"]["scoring"],
            )

            grid.fit(preprocessed_X_train, y_train)

            logging.info(f"{model_name} trained successfully")

            best_model = grid.best_estimator_

            models.append(best_model)

            logging.info(f"{model_name} appended to the list successfully.")

        return models

    except Exception as e:
        raise CustomException(e, sys)
