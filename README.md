[![Build Status](https://travis-ci.org/Nikolay-Lysenko/dsawl.svg?branch=master)](https://travis-ci.org/Nikolay-Lysenko/dsawl)
[![codecov](https://codecov.io/gh/Nikolay-Lysenko/dsawl/branch/master/graph/badge.svg)](https://codecov.io/gh/Nikolay-Lysenko/dsawl)
[![Maintainability](https://api.codeclimate.com/v1/badges/98fc23b8b51fb20f2920/maintainability)](https://codeclimate.com/github/Nikolay-Lysenko/dsawl/maintainability)

# dsawl

## What is it?
This is a set of tools for machine learning. As of now, the provided utilities look as follows:

Subject | Description | Docs
:-----: | :---------: | :--:
Active Learning | Highly-modular system that recommends which previously unlabelled examples should be labelled in order to increase model quality quickly and significantly. Special features: various options for both exploitation and exploration. | [Read more](https://github.com/Nikolay-Lysenko/dsawl/blob/master/docs/active_learning_demo.ipynb)
Stacking | A method that applies machine learning algorithm to out-of-fold predictions or transformations made by other machine learning models. Special features: support of any `sklearn`-compatible estimators (in particular, pipelines). | [Read more](https://github.com/Nikolay-Lysenko/dsawl/blob/master/docs/stacking_demo.ipynb)
Target Encoding | An alternative to one-hot encoding and hashing trick that attempts to have both memory efficiency and incorporation of all useful information from initial features. Special features: `sklearn`-compatible wrapper that can transform data out-of-fold and apply an estimator to the result.| [Read more](https://github.com/Nikolay-Lysenko/dsawl/blob/master/docs/target_encoding_demo.ipynb)

Repository name is a combination of three words: DS, saw, and awl. DS is as an abbreviation for Data Science and the latter two words represent useful tools.


## How to install the package?
The package is compatible with Python 3.5 or newer. A virtual environment where it is guaranteed that the package works can be created based on [the file](https://github.com/Nikolay-Lysenko/dsawl/blob/master/requirements.txt) named `requirements.txt`.

To install a stable release of the package, run this command:
```
pip install dsawl
```

To install the latest version from sources, execute this from your terminal:
```
cd path/to/your/destination
git clone https://github.com/Nikolay-Lysenko/dsawl
cd dsawl
pip install -e .
```

If you have any troubles with installation, your questions are welcome. 
