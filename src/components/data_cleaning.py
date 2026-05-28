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
# --- 2. Final Execution ---
# =================================================================================


def data_cleaner(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the data cleaning steps.
    """

    try:
        dataframe = refactor_date_feature(dataframe)

        logging.info(f"Data cleaning completed: {dataframe.shape}")

        return dataframe

    except Exception as e:
        raise CustomException(e, sys)
