import pandas as pd
import numpy as np
import sys
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from us_visa.entity.estimator import TargetValueMapping
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)
from us_visa.constants import SCHEMA_CONFIG_FILE_PATH, TARGET_COLUMN, CURRENT_YEAR
from us_visa.utils.main_utils import (
    read_yaml_files,
    drop_columns,
    save_object,
    save_numpy_array_data,
)

from us_visa.logger import logging
from us_visa.exception import USvisaException


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> None:
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_files(file_path=SCHEMA_CONFIG_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:

        return pd.read_csv(file_path)

    def get_data_transformer_object(self) -> Pipeline:
        logging.info(
            "Entered get_data_transformer_object method from DataTransformation class"
        )
        try:
            logging.info("Getting preprocessing from sklearn")

            numerical_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            or_transformer = OrdinalEncoder()

            logging.info("Initializing StandardScaler,OneHotEncoder,OrdinalEncoder")

            logging.info("Getting columns from schema config yaml file")

            numerical_features = self._schema_config["num_features"]
            oh_columns = self._schema_config["oh_columns"]
            or_columns = self._schema_config["or_columns"]
            transform_columns = self._schema_config["transform_columns"]

            logging.info("Initializing Power Transformer")
            transform_pipe = Pipeline(
                steps=[("transformer", PowerTransformer(method="yeo-johnson"))]
            )

            logging.info("Creating preprocessor object from ColumnTransformer")
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("OrdinalEncoder", or_transformer, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numerical_transformer, numerical_features),
                ]
            )
            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info(
                "Exited get_data_transformer_object method from DataTransformation class"
            )
            return preprocessor

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("starting initiate data transformation")

                # TRAIN
                train_df = DataTransformation.read_data(
                    self.data_ingestion_artifact.train_file_path
                )

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info("Got train features and test features of Training dataset")

                input_feature_train_df["company_age"] = (
                    CURRENT_YEAR - input_feature_train_df["yr_of_estab"]
                )
                logging.info("Added company_age column to the Training dataset")

                drop_cols = self._schema_config["drop_columns"]

                input_feature_train_df = drop_columns(
                    df=input_feature_train_df, columns=drop_cols
                )
                logging.info("drop the columns in drop_cols of Training dataset")

                target_feature_train_df = target_feature_train_df.replace(
                    TargetValueMapping()._asdict()
                )

                # TEST
                test_df = DataTransformation.read_data(
                    self.data_ingestion_artifact.test_file_path
                )

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]
                logging.info("Got train features and test features of Testing dataset")

                input_feature_test_df["company_age"] = (
                    CURRENT_YEAR - input_feature_test_df["yr_of_estab"]
                )
                logging.info("Added company_age column to the Testing dataset")

                input_feature_test_df = drop_columns(
                    df=input_feature_test_df, columns=drop_cols
                )
                logging.info("drop the columns in drop_cols of Testing dataset")

                target_feature_test_df = target_feature_test_df.replace(
                    TargetValueMapping()._asdict()
                )

                # PREPROCESSOR
                logging.info("Got the preprocessor object")
                preprocessor = self.get_data_transformer_object()

                logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )
                input_feature_train_arr = preprocessor.fit_transform(
                    input_feature_train_df
                )

                logging.info(
                    "Used the preprocessor object to fit transform the train features"
                )
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)
                logging.info(
                    "Used the preprocessor object to transform the test features"
                )
                smt = SMOTEENN(sampling_strategy="minority")

                # APPLYING SMOOTEEN TO TRAIN DATA
                logging.info("Applying SMOTEENN on Training dataset")
                input_feature_train_final, target_feature_train_final = (
                    smt.fit_resample(input_feature_train_arr, target_feature_train_df)
                )

                # APPLYING SMOOTEEN TO TEST DATA
                logging.info("Applying SMOTEENN on Testing dataset")
                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr, target_feature_test_df
                )
                logging.info("Applied SMOTEENN on training dataset")

                logging.info("Applying SMOTEENN on testing dataset")

                # CONCAT TRAIN
                logging.info("Concatinating train final and target train final")
                train_arr = np.c_[
                    input_feature_train_final, np.array(target_feature_train_final)
                ]

                # CONCAT TEST
                logging.info("Concatinating test final and target test final")
                test_arr = np.c_[
                    input_feature_test_final, np.array(target_feature_test_final)
                ]

                logging.info("Saved the preprocessor object")
                save_object(
                    self.data_transformation_config.transformed_obj_file_path,
                    preprocessor,
                )
                logging.info("Saved TRAINED numpy array")
                save_numpy_array_data(
                    self.data_transformation_config.transformed_train_file_path,
                    train_arr,
                )
                logging.info("Saved TEST numpy array")
                save_numpy_array_data(
                    self.data_transformation_config.transformed_test_file_path, test_arr
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_obj_file_path=self.data_transformation_config.transformed_obj_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                )
                logging.info(
                    f"data_transformation artifact: {data_transformation_artifact}"
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.validation_error_message)

        except Exception as e:
            raise USvisaException(e, sys)
