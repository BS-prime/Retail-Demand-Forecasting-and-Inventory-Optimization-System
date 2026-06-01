# import modules
import sys
from pathlib import Path

from src.logger import logging
from src.exception import CustomException
from src.config import load_config

# import libraries
import pandas as pd

# initialize config
config = load_config()

# locate root
ROOT_DIR = Path(__file__).resolve().parents[2]

# =================================================================================================
# --- 1. Create features out of Date feature ---
# =================================================================================================


def create_time_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Helper to create new features out of Date feature.
    """
    try:
        dataframe["Year"] = dataframe["Date"].dt.year
        dataframe["Weekday"] = dataframe["Date"].dt.day_name()
        dataframe["Day"] = dataframe["Date"].dt.day
        dataframe["Month"] = dataframe["Date"].dt.month

        logging.info(f"Created new time features: {dataframe.shape}")

        return dataframe

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 2. Create discounted feature ---
# =================================================================================================


def calculate_discounted_price(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Helper to create the discounted price feature.
    """
    try:
        dataframe["DiscountedPrice"] = dataframe["Price"] * (
            1 - dataframe["Discount"] / 100
        )

        logging.info(f"Create discounted price feature: {dataframe.shape}")

        return dataframe

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 3. Drop leakage columns ---
# =================================================================================================


def drop_leakage_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Helper to drop leakage columns.
    """
    try:
        leakage_columns = ["Units Sold", "Units Ordered", "Inventory Level"]
        dataframe = dataframe.drop(columns=leakage_columns, errors="ignore")

        logging.info(f"Feature engineering completed: {dataframe.columns}")

        return dataframe

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================
# --- 4. Set Index to Date ---
# =================================================================================


def set_index_to_date(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Helper to set the dataframe index to Date column.
    """
    try:
        dataframe = dataframe.reset_index(drop=True)
        dataframe = dataframe.set_index("Date")
        dataframe = dataframe.sort_index()

        logging.info(f"Set the dataframe index to Date column: {dataframe.index.name}")

        return dataframe

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================================
# --- 5. Create final features ---
# =================================================================================================


def feature_engineering(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all the helpers to perform feature engineering.
    Includes dropping leakage columns that would not be available at prediction time.
    """
    try:
        logging.info("Commencing feature engineering...")

        dataframe = create_time_feature(dataframe=dataframe)
        dataframe = calculate_discounted_price(dataframe=dataframe)
        dataframe = drop_leakage_columns(dataframe=dataframe)
        dataframe = set_index_to_date(dataframe=dataframe)

        logging.info("Feature engineering completed")

        return dataframe

    except Exception as e:
        raise CustomException(e, sys)
