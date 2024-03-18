from setuptools import find_packages, setup
from typing import List


def get_requirements() -> List[str]:
    """
    This function return a list of requirements
    """
    requirement_list: List[str] = []

    # Write a code to read the requirements.txt file and append each requirements in requirement_list variable.

    return requirement_list


setup(
    name="us-visa",
    version="0.0.0",
    author="Farhan",
    author_email="thinkingdatascience@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)
