from us_visa.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
)
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)

from us_visa.components.data_ingestion import DataIngestion
from us_visa.components.data_validation import DataValidation
from us_visa.components.data_transformation import DataTransformation

import sys
from us_visa.exception import USvisaException
from us_visa.logger import logging


class TrainingPipeline:
    def __init__(self) -> None:
        try:
            self.data_ingestion_config = DataIngestionConfig()
            self.data_validation_config = DataValidationConfig()
            self.data_transformation_config = DataTransformationConfig()
        except Exception as e:
            raise USvisaException(e, sys)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info(
                "Entering start_data_ingestion method of TrainingPipeline class"
            )

            logging.info("Getting data from MongoDB")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Got the train and test data from MongoDB")

            logging.info(
                "Exiting start_data_ingestion method of TrainingPipeline class"
            )
            return data_ingestion_artifact

        except Exception as e:
            raise USvisaException(e, sys)

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logging.info(
                "Entered the start_data_validation method of TrainPipeline class"
            )
            data_validation = DataValidation(
                data_validation_config=self.data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact,
            )

            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Performed data validation operation")

            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )

            return data_validation_artifact
        except Exception as e:
            raise USvisaException(e, sys)

    def start_data_transformation(
        self,
        data_validation_artifact: DataValidationArtifact,
        Data_ingestion_artifact: DataIngestionArtifact,
    ) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.data_transformation_config,
                data_ingestion_artifact=Data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )

            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            return data_transformation_artifact
        except Exception as e:
            raise USvisaException(e, sys)

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact,
                Data_ingestion_artifact=data_ingestion_artifact,
            )
        except Exception as e:
            raise USvisaException(e, sys)
