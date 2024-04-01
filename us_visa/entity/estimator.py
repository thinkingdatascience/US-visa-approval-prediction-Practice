import sys

from sklearn.pipeline import Pipeline
from pandas import DataFrame

from us_visa.exception import USvisaException
from us_visa.logger import logging


class TargetValueMapping:
    def __init__(self) -> None:
        self.Certified: int = 0
        self.Denied: int = 1

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self._asdict()

        return dict(zip(mapping_response.values(), mapping_response.keys()))


class USVisaModel:
    def __init__(
        self, preprocessing_object: Pipeline, trained_model_object: object
    ) -> None:
        try:
            self.preprocessing_object = preprocessing_object
            self.trained_model_object = trained_model_object
        except Exception as e:
            raise USvisaException(e, sys)

    def predict(self, dataframe: DataFrame) -> DataFrame:
        try:
            transformed_features = self.preprocessing_object.transform(dataframe)

            return self.trained_model_object.predict(transformed_features)

        except Exception as e:
            raise USvisaException(e, sys)
