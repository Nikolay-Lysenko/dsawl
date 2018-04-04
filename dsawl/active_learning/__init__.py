"""
The `active_learning` module contains tools for active learning.
"""


from .scorers import (
    BaseScorer, UncertaintyScorerForClassification, CommitteeScorer,
    VarianceScorerForRegression, DensityScorer, RandomScorer
)
from .pool_based_sampling import CombinedSamplerFromPool

from . import scoring_functions
from . import scorers
from . import pool_based_sampling
from . import utils


__all__ = [
    "BaseScorer", "UncertaintyScorerForClassification", "CommitteeScorer",
    "VarianceScorerForRegression", "DensityScorer", "RandomScorer",
    "CombinedSamplerFromPool",
    "scoring_functions", "scorers", "pool_based_sampling", "utils"
]
