from sklearn.pipeline import Pipeline
from pandas import DataFrame
import sys
from us_visa.exception import USVisaException
from us_visa.logger import logging


class TargetValueMapping:
    def __init__(self) -> None:
        self.Certified: int = 0
        self.Denied: int = 1

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self._asdict

        return dict(zip(mapping_response.values(), mapping_response.keys()))


class USvisaModel:
    def __init__(
        self, preprocessing_object: Pipeline, trained_model_object: object
    ) -> DataFrame:
        try:
            self.preprocessing_object = preprocessing_object
            self.trained_model_object = trained_model_object
        except Exception as e:
            raise USVisaException(e, sys)

    def predict(self, dataframe: DataFrame) -> DataFrame:
        try:
            logging.info("Entered USvisaModel()")
            transformed_features = self.preprocessing_object.transform(dataframe)

            logging.info("Used the trained model to get the prediction")
            return self.trained_model_object.predict(transformed_features)

        except Exception as e:
            raise USVisaException(e, sys)
