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
# --- 3. Create sell through rate feature ---
# =================================================================================================


def calculate_sell_through_rate(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Helper to calculate the sell through rate feature.
    """
    try:
        dataframe["SellThroughRate"] = (
            dataframe["Units Sold"] / dataframe["Inventory Level"]
        )

        logging.info(f"Create SellThroughRate feature: {dataframe.shape}")

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
    """
    try:
        logging.info("Commencing feature engineering...")

        dataframe = create_time_feature(dataframe=dataframe)
        dataframe = calculate_discounted_price(dataframe=dataframe)
        dataframe = calculate_sell_through_rate(dataframe=dataframe)
        dataframe = set_index_to_date(dataframe=dataframe)

        logging.info(f"Feature engineering completed: {dataframe.columns}")

        return dataframe

    except Exception as e:
        raise CustomException(e, sys)
