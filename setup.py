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
    version='0.1',
    description='A set of tools for machine learning',
    long_description=long_description,
    url='https://github.com/Nikolay-Lysenko/dsawl',
    author='Nikolay Lysenko',
    author_email='nikolay.lysenko.1992@gmail.com',
    license='MIT',
    keywords='active_learning categorical_features feature_engineering',
    packages=find_packages(exclude=['docs', 'tests', 'ci']),
    python_requires='>=3.5',
    install_requires=['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib']
)
