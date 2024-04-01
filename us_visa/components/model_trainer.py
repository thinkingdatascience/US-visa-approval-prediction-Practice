import sys
import numpy as np
from typing import Tuple

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from neuro_mf import ModelFactory
from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.entity.artifact_entity import (
    DataTransformationArtifact,
    ClassificationMetricsArtifact,
    ModelTrainerArtifact,
)
from us_visa.constants import MODEL_CONFIG_FILE_PATH
from us_visa.utils.main_utils import load_numpy_array_data, load_object, save_object
from us_visa.entity.estimator import USVisaModel

from us_visa.exception import USvisaException
from us_visa.logger import logging


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> None:
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise USvisaException(e, sys)

    def get_object_model_and_report(
        self, train: np.array, test: np.array
    ) -> Tuple[object, object]:
        logging.info("Entered get_object_model_and_report method of ModelTrainer class")
        try:
            logging.info("Using neuro_mf to get best model and report")
            model_factory = ModelFactory(
                model_config_path=self.model_trainer_config.model_trainer_config_file_path
            )

            X_train, X_test, y_train, y_test = (
                train[:, :-1],
                test[:, :-1],
                train[:, -1],
                test[:, -1],
            )

            best_model_details = model_factory.get_best_model(
                X=X_train,
                y=y_train,
                base_accuracy=self.model_trainer_config.expected_accuracy,
            )

            model_obj = best_model_details.best_model

            y_pred = model_obj.predict(X_test)

            accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
            precision = precision_score(y_true=y_test, y_pred=y_pred)
            f1 = f1_score(y_true=y_test, y_pred=y_pred)
            recall = recall_score(y_true=y_test, y_pred=y_pred)

            metric_artifact = ClassificationMetricsArtifact(
                accuracy_score=accuracy,
                precision_score=precision,
                f1_score=f1,
                recall_score=recall,
            )

            return best_model_details, metric_artifact

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            best_model_details, metric_artifact = self.get_object_model_and_report(
                train=train_arr, test=test_arr
            )

            if (
                best_model_details.best_score
                < self.model_trainer_config.expected_accuracy
            ):
                logging.info("No best model found with score more than expected score")
                raise Exception(
                    "No best model found with score more than expected score"
                )

            preprocessor = load_object(
                self.data_transformation_artifact.transformed_obj_file_path
            )

            usvisa_model = USVisaModel(
                preprocessing_object=preprocessor,
                trained_model_object=best_model_details.best_model,
            )

            logging.info("Created usvisa model object with preprocessor and model")

            logging.info("Created best model file path.")
            save_object(self.model_trainer_config.trained_model_file_path, usvisa_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise USvisaException(e, sys)
