import sys
from pandas import DataFrame

from us_visa.entity.config_entity import USvisaPredictorConfig
from us_visa.aws_cloud_storage.aws_s3_estimator import USvisaEstimator

from us_visa.exception import USvisaException
from us_visa.logger import logging


class USvisaData:
    def __init__(
        self,
        continent,
        education_of_employee,
        has_job_experience,
        requires_job_training,
        no_of_employees,
        region_of_employment,
        prevailing_wage,
        unit_of_wage,
        full_time_position,
        company_age,
    ) -> None:
        try:
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.company_age = company_age

        except Exception as e:
            raise USvisaException(e, sys)

    def get_us_visa_data_as_dict(self):
        logging.info("Entered get_usvisa_data_as_dict method as USvisaData class")
        try:
            input_data = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age],
            }

            logging.info("Created usvisa data dict")

            logging.info("Exited get_usvisa_data_as_dict method as USvisaData class")

            return input_data

        except Exception as e:
            raise USvisaException(e, sys)

    def get_usvisa_input_data_frame(self) -> DataFrame:
        try:
            usvisa_input_dict = self.get_us_visa_data_as_dict()

            return DataFrame(usvisa_input_dict)
        except Exception as e:
            raise USvisaException(e, sys)


class USvisaClassifier:
    def __init__(
        self,
        prediction_pipeline_config: USvisaPredictorConfig = USvisaPredictorConfig(),
    ) -> None:
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise USvisaException(e, sys)

    def predict(self, dataframe) -> str:
        try:
            logging.info("Entered predict method of USvisaClassifier class")
            model = USvisaEstimator(
                self.prediction_pipeline_config.model_bucket_name,
                self.prediction_pipeline_config.model_file_path,
            )

            result = model.predict(dataframe=dataframe)

            return result

        except Exception as e:
            raise USvisaException(e, sys)
