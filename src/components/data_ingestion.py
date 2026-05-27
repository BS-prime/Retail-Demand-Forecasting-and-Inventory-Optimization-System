# import modules
import sys
from pathlib import Path

from src.logger import logging
from src.exception import CustomException
from src.config import load_config


# import libraries
import pandas as pd

# initiate the config
config = load_config()

# locate root
ROOT_DIR = Path(__file__).resolve().parents[2]


def csv_loader(
    file_path: Path = ROOT_DIR / Path(config["input"]["path"]),
) -> pd.DataFrame:
    """
    load csv file into a dataframe.
    """

    try:
        dataframe = pd.read_csv(file_path)

        logging.info(f"Data ingestion completed from: {file_path}")

        return dataframe

    except Exception as e:
        raise CustomException(e, sys)
