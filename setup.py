"""
Just a regular `setup.py` file.

@author: Nikolay Lysenko
"""


import os
from setuptools import setup, find_packages


current_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(current_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dsawl',
    version='0.1a',
    description='A set of tools for machine learning',
    long_description=long_description,
    url='https://github.com/Nikolay-Lysenko/dsawl',
    author='Nikolay Lysenko',
    author_email='nikolay-lysenco@yandex.ru',
    license='MIT',
    keywords='machine_learning feature_engineering categorical_features',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=['numpy', 'scikit-learn', 'pandas']
)
