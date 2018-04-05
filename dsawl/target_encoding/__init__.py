"""
The `target_encoding` module contains tools related to a trick named
target encoding.
"""


from .target_encoder import TargetEncoder
from .estimators import (
    OutOfFoldTargetEncodingClassifier, OutOfFoldTargetEncodingRegressor
)

from . import target_encoder
from . import estimators


__all__ = [
    "TargetEncoder",
    "OutOfFoldTargetEncodingClassifier", "OutOfFoldTargetEncodingRegressor",
    "target_encoder", "estimators"
]
