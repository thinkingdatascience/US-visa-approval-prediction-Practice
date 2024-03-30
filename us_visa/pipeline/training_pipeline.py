from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact

from us_visa.components.data_ingestion import DataIngestion

import sys
from us_visa.exception import USvisaException
from us_visa.logger import logging


class TrainingPipeline:
    def __init__(self) -> None:
        try:
            self.data_ingestion_config = DataIngestionConfig()
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

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()

        except Exception as e:
            raise USvisaException(e, sys)
