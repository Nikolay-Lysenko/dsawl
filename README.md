[![Build Status](https://travis-ci.org/Nikolay-Lysenko/dsawl.svg?branch=master)](https://travis-ci.org/Nikolay-Lysenko/dsawl)
[![codecov](https://codecov.io/gh/Nikolay-Lysenko/dsawl/branch/master/graph/badge.svg)](https://codecov.io/gh/Nikolay-Lysenko/dsawl)
[![Maintainability](https://api.codeclimate.com/v1/badges/98fc23b8b51fb20f2920/maintainability)](https://codeclimate.com/github/Nikolay-Lysenko/dsawl/maintainability)

# dsawl

## What is it?
This is a set of tools for machine learning. As of now, the list of implemented utilities is as follows:
* Stacking, a method that applies machine learning algorithm to out-of-fold predictions or transformations made by other machine learning models. Implementation from this repository supports any `sklearn`-compatible estimators (in particular, pipelines). Some additional details are explained in [a short demo](https://github.com/Nikolay-Lysenko/dsawl/blob/master/docs/stacking_demo.ipynb).
* Target encoding, an alternative to one-hot encoding and hashing trick that attempts to have both memory efficiency and incorporation of all useful information from initial features. More details can be found in [a tutorial](https://github.com/Nikolay-Lysenko/dsawl/blob/master/docs/target_encoding_demo.ipynb).
* Active learning, a set of methods that recommend which previously unlabelled examples should be labelled in order to increase model quality both quickly and significantly. [A brief demo](https://github.com/Nikolay-Lysenko/dsawl/blob/master/docs/active_learning_demo.ipynb) is provided.

Repository name is a combination of three words: DS, saw, and awl. DS is as an abbreviation for Data Science and the latter two words represent useful tools.


## How to install the package?
The package is compatible with Python 3.5 or newer. Other requirements are installed by `pip` if you follow below instructions.

If you are using Linux, execute this from your terminal:
```
cd path/to/your/destination
git clone https://github.com/Nikolay-Lysenko/dsawl
cd dsawl
source activate your_virtual_env
pip install .
```

Instructions for other operating systems will be released in the future (probably, above commands are valid not only on Linux).

If you have any troubles with installation, your questions are welcome. A virtual environment where it is guaranteed that the package works can be created based on [the file](https://github.com/Nikolay-Lysenko/dsawl/blob/master/package-list.txt) named `package-list.txt` from the repository of the project.

After successful installation, you can use this package like any other regular package from your currently activated virtual environment. For example, you can import a class from it:
```
from dsawl.target_encoding.estimators import OutOfFoldTargetEncodingRegressor
```
