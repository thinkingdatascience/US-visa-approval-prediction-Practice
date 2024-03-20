import sys
from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact
from us_visa.components.data_ingestion import DataIngestion


from us_visa.exception import USVisaException


class TrainingPipeline:
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            # Instantiating a class
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            return data_ingestion_artifact

        except Exception as e:
            raise USVisaException(e, sys)

    def run_pipeline(self):
        try:
            data_ingestion = self.start_data_ingestion()

        except Exception as e:
            raise USVisaException(e, sys)
