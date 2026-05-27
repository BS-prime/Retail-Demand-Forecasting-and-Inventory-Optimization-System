import sys
from pathlib import Path

from src.config import load_config
from src.exception import CustomException
from src.logger import logging

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# initialize config
config_file = load_config()

# locate root
ROOT_DIR = Path(__file__).resolve().parents[2]

# =================================================================================================
# --- 1. Perform train test split ---
# =================================================================================================


def input_output_split(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Helper to split features into input and output feature, then perform
    train test split
    """
    try:
        X = dataframe.drop(columns=["Demand", "Date"], errors="ignore")
        y = dataframe["Demand"]

        logging.info(f"Input output split done: {X.shape}, {y.shape}")

        return X, y

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 2. Perform train test split ---
# =================================================================================================


def perform_train_test_split(
    X: pd.DataFrame, y: pd.Series, test_size: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Helper to perform a chronological train test split.
    """

    try:
        split_test_size = test_size or config_file["training"]["test_size"]
        split_index = int((1 - split_test_size) * len(X))

        X_train = X.iloc[:split_index].copy()
        y_train = y.iloc[:split_index].copy()
        X_test = X.iloc[split_index:].copy()
        y_test = y.iloc[split_index:].copy()

        logging.info(
            f"Train test split done: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}"
        )

        return X_train, X_test, y_train, y_test

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 3. select datatype column ---
# =================================================================================================


def datatype_based_feature_selection(
    X_train: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Helper to select features based on datatype.
    """
    try:
        num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

        logging.info(
            f"Datatype based feature selection done: {len(num_cols)} numerical features, {len(cat_cols)} categorical features"
        )

        return num_cols, cat_cols

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 4. create preprocessing pipelines ---
# =================================================================================================


def create_pipelines() -> tuple[Pipeline, Pipeline]:
    """
    Helper to create pipelines.
    """

    try:
        num_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        logging.info(f"Steps of numerical pipeline: {num_pipeline.steps}")

        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
            ]
        )

        logging.info(f"Steps of categorical pipeline: {cat_pipeline.steps}")

        return num_pipeline, cat_pipeline

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 5. Create preprocessor ---
# =================================================================================================


def create_preprocessor(
    num_pipeline: Pipeline,
    cat_pipeline: Pipeline,
    num_cols: list[str],
    cat_cols: list[str],
) -> ColumnTransformer:
    """
    Helper to create a preprocessor(ColumnTransformer).
    """

    try:
        preprocessor = ColumnTransformer(
            [("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)]
        )

        logging.info(f"Preprocessor Created: {preprocessor}")

        return preprocessor

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 5. Create preprocessor ---
# =================================================================================================


def data_transformation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    preprocessor: ColumnTransformer,
):
    """
    Helper to transform the data using preprocessor.
    """

    try:
        preprocessed_X_train = preprocessor.fit_transform(X_train)
        preprocessed_X_test = preprocessor.transform(X_test)

        logging.info(
            f"Data transformation done: {preprocessed_X_train.shape}, {preprocessed_X_test.shape}"
        )

        return preprocessed_X_train, preprocessed_X_test, preprocessor

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 6. Save feautures ---
# =================================================================================================


def save_features_to_csv(
    preprocessed_X_train: pd.DataFrame,
    preprocessed_X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config_data: dict = config_file,
) -> None:
    """
    Helper to save transformed features to CSV files.
    """
    try:
        train_arr = np.c_[preprocessed_X_train, np.array(y_train)]
        test_arr = np.c_[preprocessed_X_test, np.array(y_test)]

        logging.info(f"training features created: {train_arr.shape}")
        logging.info(f"testing features created: {test_arr.shape}")

        # save training features
        train_dir = ROOT_DIR / Path(config_data["output"]["train_feature_path"]).parent
        train_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(train_arr).to_csv(
            train_dir / "train.csv", index=False, header=False
        )

        # save testing features
        test_dir = ROOT_DIR / Path(config_data["output"]["test_feature_path"]).parent
        test_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(test_arr).to_csv(test_dir / "test.csv", index=False, header=False)

        logging.info(
            f"Transformed features saved to: {config_data['output']['train_feature_path']}, {config_data['output']['test_feature_path']}"
        )

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 7. Final execution ---
# =================================================================================================


def data_preprocessing(
    dataframe: pd.DataFrame,
):
    """
    Main function to execute data preprocessing steps.
    """

    try:
        X, y = input_output_split(dataframe)

        X_train, X_test, y_train, y_test = perform_train_test_split(X, y)

        num_cols, cat_cols = datatype_based_feature_selection(X_train)

        num_pipeline, cat_pipeline = create_pipelines()

        preprocessor = create_preprocessor(
            num_pipeline, cat_pipeline, num_cols, cat_cols
        )

        preprocessed_X_train, preprocessed_X_test, preprocessor = data_transformation(
            X_train, X_test, preprocessor
        )

        save_features_to_csv(preprocessed_X_train, preprocessed_X_test, y_train, y_test)

        return preprocessor, preprocessed_X_train, preprocessed_X_test, y_train, y_test

    except Exception as e:
        raise CustomException(e, sys)
