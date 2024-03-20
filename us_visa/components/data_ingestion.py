import os, sys
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from us_visa.entity.config_entity import DataIngestionConfig, COLLECTION_NAME
from us_visa.entity.artifact_entity import DataIngestionArtifact

from us_visa.data_access.usvisa_data import USvisaData

from us_visa.logger import logging
from us_visa.exception import USVisaException


class DataIngestion:
    def __init__(
        self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        try:

            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise USVisaException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        try:
            logging.info("Exporting data from MongoDB")
            usvisa_data = USvisaData()
            dataframe = usvisa_data.export_collection_as_dataframe(
                collection_name=COLLECTION_NAME
            )
            logging.info(f"Shape of the dataframe is {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(
                f"Saving exported file to feature store path: {feature_store_file_path}"
            )

            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            return dataframe

        except Exception as e:
            raise USVisaException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame):
        logging.info("Enterering split data into train and test method")
        try:

            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the data frame")
            logging.info("Exiting split data into train and test method")
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
            raise USVisaException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Enterering Initiate data ingestion class")

            dataframe = self.export_data_into_feature_store()
            logging.info("Got the data from mongoDB")

            self.split_data_as_train_test(dataframe=dataframe)
            logging.info("Performed train and test split on the dataset")
            logging.info("Exited Initiate data ingestion class")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )

            logging.info(f"Data Ingestion artifact:{data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise USVisaException(e, sys)
