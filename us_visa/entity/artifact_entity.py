from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    drift_report_file_path: str
    validation_status: bool
    validation_error_message: str


@dataclass
class DataTransformationArtifact:
    transformed_obj_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass
class ClassificationMetricsArtifact:
    accuracy_score: float
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metric_artifact: ClassificationMetricsArtifact


@dataclass
class ModelEvaluationArtifact:
    s3_model_path: str
    trained_model_path: str
    changed_accuracy: float
    is_model_accepted: bool


@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_file_path: str
