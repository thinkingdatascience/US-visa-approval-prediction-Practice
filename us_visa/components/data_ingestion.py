import os
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact
from us_visa.data_access.usvisa_data import USvisaData
from us_visa.constants import COLLECTION_NAME

import sys
from us_visa.logger import logging
from us_visa.exception import USvisaException


class DataIngestion:
    def __init__(
        self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ) -> None:
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise USvisaException(e, sys)

    def export_data_to_feature_store(self) -> DataFrame:
        try:
            logging.info(
                "Entered export_data_to_feature_store method of DataIngestion class"
            )
            usvisa_data = USvisaData()

            dataframe = usvisa_data.export_collection_as_dataframe(
                collection_name=COLLECTION_NAME,
            )
            logging.info(f"Shape of Dataframe: {dataframe.shape}")

            logging.info("Creating feature store file path")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(
                f"Saving exported data into feature store file path {feature_store_file_path}"
            )
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            return dataframe
        except Exception as e:
            raise USvisaException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame):
        try:
            logging.info(
                "Entered split data as trained and test method of DataIngestion class"
            )
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=0,
            )
            logging.info("performed train and test split ratio")

            logging.info(
                "Exited split data as trained and test method of DataIngestion class"
            )

            logging.info("Creating ingested file path")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train and test file path")
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            logging.info("Exported train and test file path")

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info(
                "Entered initiate_data_ingestion method of DataIngestion class"
            )
            dataframe = self.export_data_to_feature_store()

            self.split_data_as_train_test(dataframe)

            logging.info("Exited initiate_data_ingestion method of DataIngestion class")

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )

            logging.info(f"data_ingestion_artifact {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise USvisaException(e, sys)
