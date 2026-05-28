# import modules
import sys
from pathlib import Path
from typing import Any

from src.config import load_config
from src.exception import CustomException
from src.logger import logging

# import libraries
import matplotlib
import numpy as np
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# locate root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# ===========================================================================
# --- 1. Feature Name Extraction ---
# ===========================================================================


def get_transformed_feature_names(preprocessor: Any) -> list[str]:
    """
    Get feature names created by the fitted preprocessor.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = []

        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder" or transformer == "drop":
                continue

            columns = list(columns)

            try:
                transformed_names = transformer.get_feature_names_out(columns)
                feature_names.extend(transformed_names)
            except Exception:
                feature_names.extend(columns)

        return [str(feature) for feature in feature_names]


# ===========================================================================
# --- 2. Data Sampling ---
# ===========================================================================


def sample_rows(X: Any, max_samples: int = 200) -> np.ndarray:
    """
    This function speeds up the process by limiting the dataset to a maximum
    number of rows.
    """
    X_array = np.asarray(X)

    if X_array.shape[0] <= max_samples:
        return X_array

    return X_array[:max_samples]


# ===========================================================================
# --- 3. Positive Class Explanation Selection ---
# ===========================================================================


def select_positive_class_explanation(
    explanation: shap.Explanation | list[shap.Explanation],
) -> shap.Explanation:
    """
    Helper function to extract only the positive class explanation.
    """
    if isinstance(explanation, list):
        explanation = explanation[1]

    values = explanation.values

    if isinstance(values, list):
        return values[1]

    if values.ndim == 3:
        base_values = np.asarray(explanation.base_values)
        if base_values.ndim == 2:
            base_values = base_values[:, 1]

        return shap.Explanation(
            values=values[:, :, 1],
            base_values=base_values,
            data=explanation.data,
            feature_names=explanation.feature_names,
        )

    return explanation


# ===========================================================================
# --- 4. SHAP Explanation Creation ---
# ===========================================================================


def create_shap_explanation(
    model: Any,
    X: Any,
    preprocessor: Any,
    max_samples: int = 200,
) -> shap.Explanation:
    """
    Create SHAP values for the selected model on transformed feature data.
    """
    feature_names = get_transformed_feature_names(preprocessor)
    X_sample = sample_rows(X, max_samples=max_samples)

    if len(feature_names) != X_sample.shape[1]:
        raise ValueError(
            "Feature name count does not match transformed data width: "
            f"{len(feature_names)} names vs {X_sample.shape[1]} columns."
        )

    try:
        explainer = shap.Explainer(
            model,
            X_sample,
            feature_names=feature_names,
        )
    except Exception:
        explainer = shap.Explainer(
            model.predict_proba,
            X_sample,
            feature_names=feature_names,
        )

    explanation = explainer(X_sample)

    return select_positive_class_explanation(explanation)


# ===========================================================================
# --- 5. SHAP Summary Plot Saving ---
# ===========================================================================


def save_shap_summary_plot(
    model: Any,
    X: Any,
    preprocessor: Any,
    output_path: Path | None = None,
    max_samples: int = 200,
    max_display: int = 15,
) -> Path:
    """
    Save a SHAP bar summary plot for the selected model.
    """
    try:
        if output_path is None:
            config = load_config()
            output_path = ROOT_DIR / Path(config["output"]["explainability_path"])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        explanation = create_shap_explanation(
            model=model,
            X=X,
            preprocessor=preprocessor,
            max_samples=max_samples,
        )

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            explanation,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logging.info(f"SHAP summary plot saved at: {output_path}")

        return output_path

    except Exception as e:
        raise CustomException(e, sys)
