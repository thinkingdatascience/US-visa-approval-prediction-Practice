import pandas as pd
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import f1_score

from us_visa.constants import CURRENT_YEAR, TARGET_COLUMN
from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.entity.artifact_entity import (
    ModelEvaluationArtifact,
    DataIngestionArtifact,
    ModelTrainerArtifact,
)
from us_visa.aws_cloud_storage.aws_s3_estimator import USvisaEstimator
from us_visa.entity.estimator import TargetValueMapping

import sys
from us_visa.exception import USvisaException
from us_visa.logger import logging


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    s3_best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> None:
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise USvisaException(e, sys)

    def get_s3_best_model(self) -> Optional[USvisaEstimator]:
        try:
            bucket_name = self.model_evaluation_config.bucket_name
            model_path = self.model_evaluation_config.s3_model_key_path
            usvisa_estimator = USvisaEstimator(
                bucket_name=bucket_name, model_path=model_path
            )

            if usvisa_estimator.is_model_present(model_path=model_path):
                return usvisa_estimator
            else:
                return None
        except Exception as e:
            raise USvisaException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df["company_age"] = CURRENT_YEAR - test_df["yr_of_estab"]

            X = test_df.drop(TARGET_COLUMN, axis=1)
            y = test_df[TARGET_COLUMN]

            y = y.replace(TargetValueMapping()._asdict())

            # Calling trained model f1 score
            trained_model_f1_score = (
                self.model_trainer_artifact.metric_artifact.f1_score
            )

            # Calling best model f1 score from s3 bucket if present.
            s3_best_model_f1_score = None

            s3_best_model = self.get_s3_best_model()

            if s3_best_model is not None:
                y_hat_s3_best_model = s3_best_model.predict(X)
                s3_best_model_f1_score = f1_score(y_true=y, y_pred=y_hat_s3_best_model)

            tmp_s3_best_model_f1_score = (
                0 if s3_best_model is None else s3_best_model_f1_score
            )

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                s3_best_model_f1_score=tmp_s3_best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_s3_best_model_f1_score,
                difference=trained_model_f1_score - tmp_s3_best_model_f1_score,
            )
            logging.info(f"Results: {result}")
            return result

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                s3_model_path=self.model_evaluation_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,
                is_model_accepted=evaluate_model_response.is_model_accepted,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise USvisaException(e, sys)
