[![Build Status](https://travis-ci.org/Nikolay-Lysenko/dsawl.svg?branch=master)](https://travis-ci.org/Nikolay-Lysenko/dsawl)
[![codecov](https://codecov.io/gh/Nikolay-Lysenko/dsawl/branch/master/graph/badge.svg)](https://codecov.io/gh/Nikolay-Lysenko/dsawl)
[![Code Climate](https://codeclimate.com/github/Nikolay-Lysenko/dsawl/badges/gpa.svg)](https://codeclimate.com/github/Nikolay-Lysenko/dsawl)

# dsawl

## What is it?
This is a set of tools for machine learning. As of now, the list of implemented utilities is as follows:
* Out-of-fold feature generation, an alternative to one-hot encoding that is more memory efficient if some categorical variables take a lot of distinct values. See `docs/ooffg/demo.ipynb` for more details.

Repository name is a combination of three words: DS, saw, and awl. DS is as an abbreviation for Data Science and the latter two words represent useful tools.


## How to install the package?
The package is compatible with Python 3.5 or newer. Other requirements are installed by `conda` if you follow below instructions.

If you are using Linux, execute this from your terminal:
```
cd path/to/your/destination
git clone https://github.com/Nikolay-Lysenko/dsawl
cd dsawl
source activate your_conda_env
conda install --file package-list.txt
pip install .
```

Instructions for other operating systems will be released in the future (probably, above commands are valid not only on Linux).

After successful installation, you can use this package like any other regular package from your currently activated virtual environment. For example, you can import a class from it:
```
from dsawl.ooffg.estimators import OutOfFoldFeaturesRegressor
```
