# import modules
import sys
from pathlib import Path

from src.config import load_config
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# import functions
from src.components.data_ingestion import csv_loader
from src.components.data_cleaning import data_cleaner
from src.components.feature_engineering import feature_engineering
from src.components.data_preprocessing import data_preprocessing
from src.components.model_training import model_trainer
from src.components.model_evaluation import model_evaluator
from src.components.model_explainability import save_shap_summary_plot


# locate the root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

def run_training_pipeline():
    """
    Combine all the functions to train, evaluate, and explain the model.
    """

    try:
        config_file = load_config()

        # -----------------------------------------------------------------------------------------
        # --- 1. Data Ingestion ---
        # -----------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Data Ingestion")
        df = csv_loader()

        # -----------------------------------------------------------------------------------------
        # --- 2. Data Cleaning ---
        # -----------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Data Cleaning")
        df = data_cleaner(df)

        # -----------------------------------------------------------------------------------------
        # --- 3. Feature Engineering ---
        # -----------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Feature Engineering")
        df = feature_engineering(df)

        # -----------------------------------------------------------------------------------------
        # --- 4. Data Preprocessing ---
        # -----------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Data Preprocessing")
        (
            preprocessor,
            preprocessed_X_train,
            preprocessed_X_test,
            X_train,
            X_test,
            y_train,
            y_test,
        ) = data_preprocessing(df)

        # -----------------------------------------------------------------------------------------
        # --- 5. Model Training ---
        # -----------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Model Training")
        models = model_trainer(X_train=X_train, y_train=y_train, preprocessor=preprocessor)

        # -----------------------------------------------------------------------------------------
        # --- 6. Model Evaluation ---
        # -----------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Model Evaluation")
        best_model = model_evaluator(models=models, X_test=X_test, y_test=y_test)
        if best_model is None:
            raise ValueError("Model evaluation did not select a best model.")

        fitted_preprocessor = (
            best_model.named_steps["preprocessor"]
            if hasattr(best_model, "named_steps")
            else preprocessor
        )

        # -----------------------------------------------------------------------------------------
        # --- 7. Save the preprocessor ---
        # -----------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Save preprocessor")
        save_object(filepath=config_file["output"]["preprocessor_path"], obj=fitted_preprocessor)

        logging.info(
            f"Preprocessor saved at: {config_file['output']['preprocessor_path']}"
        )

        # -----------------------------------------------------------------------------------------
        # --- 8. Save the Model ---
        # -----------------------------------------------------------------------------------------

        logging.info("Starting Pipeline: Save Model")
        save_object(filepath=config_file["output"]["model_path"], obj=best_model)

        logging.info(f"Model saved at: {config_file['output']['model_path']}")

        # -----------------------------------------------------------------------------------------
        # --- 9. Model Explainability ---
        # -----------------------------------------------------------------------------------------

        logging.info("Saving SHAP Summary Plot")
        shap_model = (
            best_model.named_steps["model"]
            if hasattr(best_model, "named_steps")
            else best_model
        )
        shap_X = (
            fitted_preprocessor.transform(X_test)
            if hasattr(best_model, "named_steps")
            else preprocessed_X_test
        )
        save_shap_summary_plot(
            model=shap_model,
            X=shap_X,
            preprocessor=fitted_preprocessor,
        )
    
    except Exception as e:
        raise CustomException(e, sys)
    
if __name__ == "__main__":
    run_training_pipeline()
