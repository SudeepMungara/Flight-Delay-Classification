from setuptools import find_packages,setup
from typing import List

def get_requirements() -> List[str]:
    """This function will return list of requirements from requirements.txt file.

    Returns:
        List[str]: _description_
    """
    requirement:List[str] = []
    try:
        with open("requirements.txt", "r") as file:
            lines =  file.readlines() #read lines from file
            for line in lines:
                requirement_line = line.strip()
                if requirement_line and requirement_line!='-e .':
                    requirement.append(requirement_line)
    except FileNotFoundError:
        print("requirements.txt file not found.")
    return requirement

setup(
    name='Flight Delay Classification',
    version='0.0.1',
    author='Sudeep Mungara',
    packages=find_packages(),
    install_requires=get_requirements(),
)

        