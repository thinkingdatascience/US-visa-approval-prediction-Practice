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
