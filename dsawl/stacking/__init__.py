"""
The `stacking` module contains tools for stacking models.
"""


from .stackers import BaseStacking, StackingClassifier, StackingRegressor

from . import stackers
from . import utils


__all__ = [
    "BaseStacking", "StackingClassifier", "StackingRegressor",
    "stackers", "utils"
]
