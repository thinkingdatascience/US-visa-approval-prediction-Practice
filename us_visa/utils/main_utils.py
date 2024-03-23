import os
import sys
import numpy as np
import dill
import yaml
from pandas import DataFrame

from us_visa.exception import USVisaException
from us_visa.logger import logging


# FOR YAML FILES
def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file=file_path, mode="rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise USVisaException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file=file_path, mode="w") as yaml_file:
            yaml.dump(content, yaml_file)

    except Exception as e:
        raise USVisaException(e, sys)


# FOR OBJECTS
def load_object(file_path: str) -> object:
    logging.info("Entered the load object method of utils")
    try:
        with open(file=file_path, mode="rb") as fileobj:
            obj = dill.load(file=fileobj)

            logging.info("Exited the load object method of utils")
            return obj

    except Exception as e:
        raise USVisaException(e, sys)


def save_object(file_path: str, object: object) -> None:
    logging.info("Entered the save object method of utils")
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file=file_path, mode="wb") as fileobj:
            dill.dump(obj=object, file=fileobj)

        logging.info("Exited the save object method of utils")

    except Exception as e:
        raise USVisaException(e, sys)


# FOR NUMPY FILES
def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file=file_path, mode="rb") as fileobj:
            return np.load(file=fileobj)

    except Exception as e:
        raise USVisaException(e, sys)


def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(name=dir_path, exist_ok=True)

        with open(file=file_path, mode="wb") as fileobj:
            np.save(file=fileobj, arr=array)

    except Exception as e:
        raise USVisaException(e, sys)


# FOR DROPPING COLUMNS
def drop_columns(df: DataFrame, columns: list) -> DataFrame:
    logging.info("Entered the drop column method of utils")
    try:
        df.drop(columns=columns, axis=1)

        logging.info("Exited the drop column method of utils")
        return df

    except Exception as e:
        raise USVisaException(e, sys)
