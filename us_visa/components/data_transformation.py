import sys
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
)

from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)
from us_visa.utils.main_utils import (
    read_yaml_file,
    drop_columns,
    save_object,
    save_numpy_array_data,
)
from us_visa.constants import SCHEMA_FILE_PATH, TARGET_COLUMN, CURRENT_YEAR

from us_visa.entity.estimator import TargetValueMapping

from us_visa.logger import logging
from us_visa.exception import USVisaException


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
        data_ingestion_artifact: DataIngestionArtifact,
    ):

        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise USVisaException(e, sys)

    logging.info("Entering Data Transformation class")

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USVisaException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            logging.info(
                "Entering get_data_transformer_object method from Data Transformation"
            )
            logging.info(
                "Initializing StandardScaler, OneHotEncoder, OrdinalEncoder from sklearn library"
            )

            numeric_transformer = StandardScaler()
            onehot_encoder = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            logging.info("Calling numerical columns from schema.yaml")

            oh_columns = self._schema_config["oh_columns"]
            or_columns = self._schema_config["or_columns"]
            transform_columns = self._schema_config["transform_columns"]
            num_features = self._schema_config["num_features"]

            logging.info("Initializing Power Transformer")
            transform_pipe = Pipeline(
                steps=[("transformer", PowerTransformer(method="yeo-johnson"))]
            )

            logging.info("Creating Preprocessor object")
            preprocessor = ColumnTransformer(
                [
                    ("OneHot_Encoder", onehot_encoder, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformers", transform_pipe, transform_columns),
                    ("Standard_scalar", numeric_transformer, num_features),
                ]
            )
            logging.info("Created Preprocessor object from Column Transformer")
            logging.info(
                "Exiting get_data_transformer_object method from Data Transformation"
            )

            return preprocessor

        except Exception as e:
            raise USVisaException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting Data Transformation")
                logging.info("Reading train and Test dataset")
                train_df = DataTransformation.read_data(
                    file_path=self.data_ingestion_artifact.trained_file_path
                )
                test_df = DataTransformation.read_data(
                    file_path=self.data_ingestion_artifact.test_file_path
                )

                logging.info("Got the preprocessor object")
                preprocessor = self.get_data_transformer_object()

                logging.info("Transforming Training Features to train and target")
                input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info(
                    "Replacing yr_of estab with company age in training dataset"
                )
                input_feature_train_df["company_age"] = (
                    CURRENT_YEAR - input_feature_train_df["yr_of_estab"]
                )
                logging.info("dropping unnacessary columns")
                drop_cols = self._schema_config["drop_columns"]

                input_feature_train_df = drop_columns(
                    df=input_feature_train_df, columns=drop_cols
                )

                logging.info("converting target column to binary in training df")
                target_feature_train_df = target_feature_train_df.replace(
                    TargetValueMapping()._asdict()
                )

                logging.info("Transforming Testing Features to train and target")
                input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                logging.info("Replacing yr_of estab with company age in test dataset")
                input_feature_test_df["company_age"] = (
                    CURRENT_YEAR - input_feature_test_df["yr_of_estab"]
                )

                logging.info("dropping unnacessary columns")
                input_feature_test_df = drop_columns(
                    df=input_feature_test_df, columns=drop_cols
                )

                logging.info("converting target column to binary in testing df")
                target_feature_test_df = target_feature_test_df.replace(
                    TargetValueMapping()._asdict()
                )

                logging.info(
                    "Used the preprocessor object to transform the train features"
                )
                input_feature_train_array = preprocessor.fit_transform(
                    input_feature_train_df
                )

                logging.info(
                    "Used the preprocessor object to transform the test features"
                )
                input_feature_test_array = preprocessor.transform(input_feature_test_df)

                logging.info("Applying SMOTEENN on Training dataset")
                smt = SMOTEENN(sampling_strategy="minority")

                logging.info("Applied SMOTEENN on training dataset")
                input_feature_train_final, target_feature_train_final = (
                    smt.fit_resample(input_feature_train_array, target_feature_train_df)
                )

                logging.info("Applied SMOTEENN on testing dataset")
                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_array, target_feature_test_df
                )

                logging.info("Created train array and test array")
                train_array = np.c_[
                    input_feature_train_final, np.array(target_feature_train_final)
                ]
                test_array = np.c_[
                    input_feature_test_final, np.array(target_feature_test_final)
                ]

                logging.info(
                    "Saving preprocessing file and numpy array files for train and test"
                )

                save_object(
                    file_path=self.data_transformation_config.transformed_object_file_path,
                    object=preprocessor,
                )

                save_numpy_array_data(
                    file_path=self.data_transformation_config.transformed_train_file_path,
                    array=train_array,
                )
                save_numpy_array_data(
                    file_path=self.data_transformation_config.transformed_test_file_path,
                    array=test_array,
                )

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )
                data_transformation_artifact = DataTransformationArtifact(
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                )

                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise USVisaException(e, sys)
