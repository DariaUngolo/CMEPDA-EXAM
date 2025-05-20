from setuptools import setup, find_packages

setup(
    name='CMEPDA_EXAM',
    version='0.0.1',
    packages=find_packages(include=['ML_codes', 'ML_codes.*', 'ML_main', 'ML_main.*']),
)