import logging
from datetime import datetime
from pathlib import Path

# 1. Locate the project root
PROJECT_DIR = Path(__file__).resolve().parents[1]

# 2. create the 'logs' DIRECTORY
LOGS_DIR = PROJECT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# 3. Define the full path to the log FILE
LOG_FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.log"
LOG_FILE_PATH = LOGS_DIR / LOG_FILE_NAME

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
