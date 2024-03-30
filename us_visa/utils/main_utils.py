import os
import sys
import yaml
from us_visa.exception import USvisaException


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
