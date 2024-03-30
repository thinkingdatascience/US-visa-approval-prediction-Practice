from datetime import datetime
import os

MONGODB_URL_KEY = "MONGODB_URL"
DATABASE_NAME = "US_VISA"
COLLECTION_NAME = "visa_data"

ARTIFACT_DIR = "artifact"
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

FILE_NAME = "usvisa.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

SCHEMA_CONFIG_FILE_PATH = os.path.join("config", "schema.yaml")


# DATA INGESTION RELATED CONSTANT START WITH DATA_INGESTION VARIABLE NAME

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# DATA VALIDATION RELATED CONSTANT START WITH VALIDATION VARIABLE NAME
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
