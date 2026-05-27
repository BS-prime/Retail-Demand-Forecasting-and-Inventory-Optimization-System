# import modules
import sys

from src.exception import CustomException
from src.logger import logging

# import libraries
import pandas as pd

# =================================================================================
# --- 1. Refactor the Date feature ---
# =================================================================================


def refactor_date_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Change datatype of Date column
    """
    try:
        dataframe["Date"] = pd.to_datetime(dataframe["Date"])

        logging.info(f"Cleaned the dataframe: {dataframe.shape}")

        return dataframe

    except Exception as e:
        raise CustomException(e, sys)


# =================================================================================
# --- 2. Set Index to Date ---
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


# =================================================================================
# --- 3. Final Execution ---
# =================================================================================


def data_cleaning(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the data cleaning steps.
    """

    try:
        dataframe = refactor_date_feature(dataframe)
        dataframe = set_index_to_date(dataframe)

        logging.info(f"Data cleaning completed: {dataframe.shape}")

        return dataframe

    except Exception as e:
        raise CustomException(e, sys)
