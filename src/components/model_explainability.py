import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import shap

from src.config import load_config
from src.exception import CustomException
from src.logger import logging

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]


def get_transformed_feature_names(preprocessor: Any) -> list[str]:
    """
    Extract feature names from the fitted preprocessor.
    """
    try:
        return [str(feature) for feature in preprocessor.get_feature_names_out()]
    except Exception as e:
        logging.warning(f"Unable to get feature names directly from preprocessor: {e}")

    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue

        columns = list(columns)
        try:
            feature_names.extend(transformer.get_feature_names_out(columns))
        except Exception:
            feature_names.extend(columns)

    return [str(feature) for feature in feature_names]


def create_shap_explanation(
    model: Any,
    X: Any,
    preprocessor: Any,
    max_samples: int = 200,
) -> shap.Explanation:
    """
    Create SHAP values for a fitted regression model.
    """
    if model is None:
        raise ValueError("Model explainability requires a fitted model, but received None.")

    if not hasattr(model, "predict"):
        raise TypeError("Model explainability requires a model with a predict method.")

    X_sample = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X_sample = X_sample[:max_samples]
    feature_names = get_transformed_feature_names(preprocessor)

    if len(feature_names) != X_sample.shape[1]:
        raise ValueError(
            "Feature name count does not match transformed data width: "
            f"{len(feature_names)} names vs {X_sample.shape[1]} columns."
        )

    explainer = shap.Explainer(
        model.predict,
        X_sample,
        feature_names=feature_names,
    )
    return explainer(X_sample)


def save_shap_summary_plot(
    model: Any,
    X: Any,
    preprocessor: Any,
    output_path: Path | str | None = None,
    max_samples: int = 200,
    max_display: int = 15,
) -> Path:
    """
    Save a SHAP bar summary plot for the best regression model.
    """
    try:
        if output_path is None:
            output_path = load_config()["output"]["explainability_path"]

        file_path = Path(output_path)
        file_path = file_path if file_path.is_absolute() else ROOT_DIR / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        explanation = create_shap_explanation(
            model=model,
            X=X,
            preprocessor=preprocessor,
            max_samples=max_samples,
        )

        plt.figure(figsize=(12, 7))
        shap.summary_plot(
            explanation,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close()

        logging.info(f"SHAP summary plot saved at: {file_path}")
        return file_path

    except Exception as e:
        plt.close()
        raise CustomException(e, sys)
