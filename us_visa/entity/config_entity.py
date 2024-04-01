import os
from dataclasses import dataclass
from us_visa.constants import *


@dataclass
class TrainingPipelineConfig:
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingested_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME
    )
    feature_store_file_path: str = os.path.join(
        data_ingested_dir, DATA_INGESTION_FEATURE_STORE, FILE_NAME
    )
    training_file_path: str = os.path.join(
        data_ingested_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME
    )
    testing_file_path: str = os.path.join(
        data_ingested_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME
    )
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME
    )
    drift_report_file_path: str = os.path.join(
        data_validation_dir,
        DATA_VALIDATION_DRIFT_REPORT_DIR,
        DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
    )


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR
    )
    transformed_obj_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
        PREPROCESSING_OBJ_FILE_NAME,
    )
    transformed_train_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        TRAIN_FILE_NAME_NUMPY,
    )
    transformed_test_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        TEST_FILE_NAME_NUMPY,
    )


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME
    )
    trained_model_file_path: str = os.path.join(
        model_trainer_dir,
        MODEL_TRAINER_TRAINED_MODEL_DIR,
        TRAINED_MODEL_FILENAME,
    )
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_trainer_config_file_path: str = MODEL_CONFIG_FILE_PATH
