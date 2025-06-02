from setuptools import setup, find_packages
import os

# Percorso al file requirements.txt
current_dir = os.path.abspath(os.path.dirname(__file__))
requirements_path = os.path.join(current_dir, 'requirements.txt')

# Legge il file requirements.txt
with open(requirements_path, 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='CMEPDA-EXAM',
    version='0.0.1',
    packages=find_packages(include=['ML_codes', 'ML_codes.*', 'ML_main', 'ML_main.*']),
    install_requires=requirements,  # Aggiunge i requisiti qui
)
