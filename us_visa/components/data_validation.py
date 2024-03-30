import json
import pandas as pd
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from us_visa.utils.main_utils import read_yaml_files, write_yaml_file
from us_visa.constants import SCHEMA_CONFIG_FILE_PATH
from us_visa.logger import logging

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

import sys
from us_visa.logger import logging
from us_visa.exception import USvisaException


class DataValidation:
    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ) -> None:
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self._schema_config = read_yaml_files(SCHEMA_CONFIG_FILE_PATH)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def validate_number_of_colunms(self, dataframe: pd.DataFrame) -> bool:
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required columns present: {status}")
            return status

        except Exception as e:
            raise USvisaException(e, sys)

    def is_columns_exists(self, df: pd.DataFrame) -> bool:
        try:
            dataframe_columns = df.columns

            # FOR NUMERICAL COLUMNS
            missing_numerical_columns = []
            for col in self._schema_config["numerical_columns"]:
                if col not in dataframe_columns:
                    missing_numerical_columns.append(col)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing Numerical columns: {missing_numerical_columns}")

            # FOR CATEGORICAL COLUMNS
            missing_categorical_columns = []
            for col in self._schema_config["categorical_columns"]:
                if col not in dataframe_columns:
                    missing_categorical_columns.append(col)

            if len(missing_categorical_columns) > 0:
                logging.info(
                    f"Missing categorical columns: {missing_categorical_columns}"
                )

            return (
                False
                if len(missing_categorical_columns) > 0
                or len(missing_numerical_columns) > 0
                else True
            )

        except Exception as e:
            raise USvisaException(e, sys)

    def detect_data_drift(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> bool:
        try:
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])

            data_drift_profile.calculate(
                reference_data=reference_df, current_data=current_df
            )

            report = data_drift_profile.json()
            json_report = json.loads(report)

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=json_report,
            )

            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"][
                "n_drifted_features"
            ]
            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]

            return drift_status

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_error_message = ""

            logging.info("Starting data validation")
            train_df = DataValidation.read_data(
                file_path=self.data_ingestion_artifact.train_file_path
            )
            test_df = DataValidation.read_data(
                file_path=self.data_ingestion_artifact.test_file_path
            )

            status = self.validate_number_of_colunms(dataframe=train_df)

            logging.info(
                f"All required columns persent in training dataframe: {status}"
            )
            if not status:
                validation_error_message += "Columns are missing in training dataframe"

            status = self.validate_number_of_colunms(dataframe=test_df)
            logging.info(f"All required columns persent in testing dataframe: {status}")

            if not status:
                validation_error_message += "Columns are missing in testing dataframe"

            status = self.is_columns_exists(df=train_df)
            if not status:
                validation_error_message += "Columns are missing in training dataframe"

            status = self.is_columns_exists(df=test_df)
            if not status:
                validation_error_message += "Columns are missing in testing dataframe"

            validation_status = len(validation_error_message) == 0

            if validation_status:
                drift_status = self.detect_data_drift(
                    reference_df=train_df, current_df=test_df
                )

                if drift_status:
                    logging.info("Drift Detected")
                    validation_error_message = "Drift Detected"
                else:
                    validation_error_message = "Drift not detected"

            else:
                logging.info(f"validation_error: {validation_error_message}")

            data_validation_artifact = DataValidationArtifact(
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
                validation_status=validation_status,
                validation_error_message=validation_error_message,
            )
            logging.info("Exiting data validation")
            logging.info(f"data validation artifact: {data_validation_artifact}")

            return data_validation_artifact
        except Exception as e:
            raise USvisaException(e, sys)
