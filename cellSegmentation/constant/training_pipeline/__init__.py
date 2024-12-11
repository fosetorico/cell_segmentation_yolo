import os

ARTIFACTS_DIR: str = "artifacts"
"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DIRECTORY_PATH = "cell_data"
DATA_DOWNLOAD_URL: str = "https://drive.google.com/file/d/1pVXbuBjQmIlB3oyV9jIPzUNnYT9BqMTv/view?usp=sharing"


"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_STATUS_FILE = 'status.txt'
DATA_VALIDATION_ALL_REQUIRED_FILES = ["train", "valid", "test", "data.yaml"]

