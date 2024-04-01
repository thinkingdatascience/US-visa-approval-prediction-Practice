import os
import sys
import yaml, dill
from pandas import DataFrame
import numpy as np
from us_visa.exception import USvisaException
from us_visa.logger import logging


# DROP COLUMNS
def drop_columns(df: DataFrame, columns: list) -> DataFrame:
    logging.info("Entering drop columns method of utils")

    try:
        df = df.drop(columns=columns, axis=1)

        logging.info("Exiting drop columns method of utils")

        return df

    except Exception as e:
        raise USvisaException(e, sys)


# YAML FILES
def read_yaml_files(file_path) -> dict:
    try:
        with open(file=file_path, mode="rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise USvisaException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False):
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file=file_path, mode="w") as yaml_file:
            yaml.dump(content, yaml_file)

    except Exception as e:
        raise USvisaException(e, sys)


# OBJECT FILES
def save_object(file_path: str, object: object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_object:
            dill.dump(object, file_object)

    except Exception as e:
        raise USvisaException(e, sys)


def load_object(file_path: str):
    logging.info("Entered the load_object method of utils")
    try:
        with open(file_path, mode="rb") as file_object:
            object = dill.load(file_object)

        logging.info("Exited the load_object method of utils")
        return object
    except Exception as e:
        raise USvisaException(e, sys)


# NUMPY FILES
def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as numpy_file:
            np.save(numpy_file, array)

    except Exception as e:
        raise USvisaException(e, sys)


def load_numpy_array_data(file_path: str):
    try:
        with open(file_path, "rb") as numpy_file:
            return np.load(numpy_file)

    except Exception as e:
        raise USvisaException(e, sys)
