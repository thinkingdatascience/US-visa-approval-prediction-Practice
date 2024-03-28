import sys
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import f1_score

import pandas as pd
from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.entity.estimator import TargetValueMapping
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)
from us_visa.aws_cloud_storage.aws_s3_estimator import USvisaEstimator
from us_visa.constants import CURRENT_YEAR, TARGET_COLUMN

from us_visa.logger import logging
from us_visa.exception import USVisaException


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise USVisaException(e, sys)

    # getting best model from s3 bucket
    def get_best_model(self) -> Optional[USvisaEstimator]:
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path

            usvisa_estimator = USvisaEstimator(
                bucket_name=bucket_name, model_path=model_path
            )

            if usvisa_estimator.is_model_present(model_path=model_path):
                return usvisa_estimator
            return None

        except Exception as e:
            raise USVisaException(e, sys)

    def evaluate_model(self):
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df["company_age"] = CURRENT_YEAR - test_df["yr_of_estab"]
            X = test_df.drop(TARGET_COLUMN, axis=1)
            y = test_df[TARGET_COLUMN]
            y = y.replace(TargetValueMapping()._asdict())

            trained_model_f1_score = (
                self.model_trainer_artifact.metric_artifact.f1_score
            )

            best_model_f1_score = None

            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict()
                best_model_f1_score = f1_score(
                    y,
                    y_hat_best_model,
                )

            tmp_best_model_score = (
                0 if best_model_f1_score is None else best_model_f1_score
            )

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                difference=trained_model_f1_score - tmp_best_model_score,
            )
            logging.info(f"results: {result}")
            return result
        except Exception as e:
            raise USVisaException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,
            )
            logging.info(f"Model Evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise USVisaException(e, sys)
