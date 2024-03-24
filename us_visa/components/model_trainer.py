import numpy as np
from typing import Tuple
import sys

from neuro_mf import ModelFactory
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from us_visa.utils.main_utils import load_numpy_array_data, load_object, save_object
from us_visa.entity.estimator import USvisaModel
from us_visa.entity.config_entity import ModelTrainerConfig, MODEL_CONFIG_FILE_PATH
from us_visa.entity.artifact_entity import (
    DataTransformationArtifact,
    ClassificationMetricArtifact,
    ModelTrainerArtifact,
)

from us_visa.exception import USVisaException
from us_visa.logger import logging


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise USVisaException(e, sys)

    def get_object_model_and_report(
        self, train: np.array, test: np.array
    ) -> Tuple[object, object]:
        try:
            logging.info("Splitting 'X' and 'y' from train and test")
            X_train, X_test, y_train, y_test = (
                train[:, :-1],
                test[:, :-1],
                train[:, -1],
                test[:, -1],
            )

            logging.info("Using neuro_mf to get the best model object and report")
            model_factory = ModelFactory(model_config_path=MODEL_CONFIG_FILE_PATH)

            best_model_details = model_factory.get_best_model(
                X=X_train,
                y=y_train,
                base_accuracy=self.model_trainer_config.expected_accuracy,
            )
            model_obj = best_model_details.best_model

            y_pred = model_obj.predict(X_test)

            accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
            f1 = f1_score(y_true=y_test, y_pred=y_pred)
            precision = precision_score(y_true=y_test, y_pred=y_pred)
            recall = recall_score(y_true=y_test, y_pred=y_pred)

            metric_artifact = ClassificationMetricArtifact(
                accuracy_score=accuracy,
                f1_score=f1,
                precision_score=precision,
                recall_score=recall,
            )

            return best_model_details, metric_artifact

        except Exception as e:
            raise USVisaException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Entered initiate_model_trainer method of ModelTrainer class")
            train_array = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_array = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            best_model_details, metric_artifact = self.get_object_model_and_report(
                train=train_array, test=test_array
            )

            if (
                best_model_details.best_score
                < self.model_trainer_config.expected_accuracy
            ):
                raise Exception("No best model found with score more than base score")

            preprocessing_object = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            usvisa_model = USvisaModel(
                preprocessing_object=preprocessing_object,
                trained_model_object=best_model_details.best_model,
            )
            logging.info("Created USvisaModel with preprocessor and model")
            logging.info("Created best model file path")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=usvisa_model,
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise USVisaException(e, sys)
