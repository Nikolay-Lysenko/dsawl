"""
This file contains highly-modular tools for pool-based approach
to active learning.

Active learning setup assumes that, given a model and a training set,
it is possible to extend the training set with new labelled examples
and the goal is to do it with maximum possible improvement of model
quality subject to constraint on how many new examples can be added.
Further, pool-bases sampling means that new examples come from
a fixed and known set of initially unlabelled examples, i.e., the task
is to choose objects to be studied, not to synthesize them arbitrarily.

@author: Nikolay Lysenko
"""


from typing import List, Union, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.mixture.base import BaseMixture

from .scoring_functions import (
    compute_confidences, compute_margins, compute_entropy,
    compute_committee_divergences, compute_committee_variances,
    compute_estimations_of_variance
)
from .scorers import (
    ToolsType, BaseScorer, UncertaintyScorerForClassification, CommitteeScorer,
    VarianceScorerForRegression, RandomScorer, DensityScorer
)


class CombinedSamplerFromPool:
    """
    This class is for selection of objects to be labelled
    in a pool-based sampling setup of active learning.

    :param scorers:
        list of scorers for ranking new objects, each its element
        also can be one of these strings: 'confidence', 'margin',
        'entropy', 'divergence', 'predictions_variance',
        'target_variance', 'random', 'density'
    :param scorers_probabilities:
        list such that its i-th element is the probability of
        selecting objects based on scores of the i-th scorer
    """

    def __init__(
            self,
            scorers: List[Union[str, BaseScorer]],
            scorers_probabilities: Optional[List[float]] = None
            ):
        self.__scorers = None
        self.__set_scorers(scorers)
        self.__scorers_probabilities = None
        scorers_probabilities = scorers_probabilities or [1 for _ in scorers]
        self.set_scorers_probabilities(scorers_probabilities)

    def __set_scorers(
            self, scorers: List[Union[str, BaseScorer]]
            ) -> type(None):
        # Set scorers based on passed values.
        str_to_scorer = {
            'confidence': UncertaintyScorerForClassification(
                compute_confidences, revert_sign=True
            ),
            'margin': UncertaintyScorerForClassification(
                compute_margins, revert_sign=True
            ),
            'entropy': UncertaintyScorerForClassification(
                compute_entropy
            ),
            'divergence': CommitteeScorer(
                compute_committee_divergences
            ),
            'predictions_variance': CommitteeScorer(
                compute_committee_variances, is_classification=False
            ),
            'target_variance': VarianceScorerForRegression(
                compute_estimations_of_variance,
            ),
            'random': RandomScorer(),
            'density': DensityScorer()
        }
        scorers = [str_to_scorer.get(scorer, scorer) for scorer in scorers]
        self.__scorers = scorers

    def set_scorers_probabilities(
            self,
            scorers_probabilities: List[float]
            ) -> type(None):
        """
        Replace probabilities of scores with a new array.
        In particular, it can be useful for gradual decreasing of
        exploratory actions probability.

        :param scorers_probabilities:
            list such that its i-th element is the probability of
            selecting objects based on scores of the i-th scorer
        :return:
            None
        """
        if len(self.__scorers) != len(scorers_probabilities):
            raise ValueError("Lengths do not match.")
        scorers_probabilities = [
            x / sum(scorers_probabilities) for x in scorers_probabilities
        ]
        self.__scorers_probabilities = scorers_probabilities

    def pick_new_objects(
            self,
            X_new: np.ndarray,
            n_to_pick: int = 1
            ) -> List[int]:
        """
        Select objects from a fixed pool of objects.

        :param X_new:
            feature representation of new objects
        :param n_to_pick:
            number of objects to pick
        :return:
            indices of the most important objects
        """
        scorer = np.random.choice(
            self.__scorers, p=self.__scorers_probabilities
        )
        scores = scorer.score(X_new)
        picked_indices = scores.argsort()[-n_to_pick:].tolist()
        return picked_indices

    def get_tools(
            self,
            scorer_id: Optional[int] = None
            ) -> Union[ToolsType, List[ToolsType]]:
        """
        Get estimator or ensemble of estimators such that it is used
        by the specified scorer(s) for scoring new objects
        by usefulness of their labels.

        :param scorer_id:
            identifier (number) of scorer; if it is not passed,
            list of tools of all scorers is returned
        :return:
            internal tools of scorer(s)
        """
        if scorer_id is not None:
            return self.__scorers[scorer_id].get_tools()
        else:
            return [scorer.get_tools() for scorer in self.__scorers]

    def set_tools(self, tools: ToolsType, scorer_id: int) -> type(None):
        """
        Replace internal tools of the specified scorer with
        the passed tools.

        :param tools:
            new internal tools of scorer
        :param scorer_id:
            identifier (number) of scorer
        :return:
            None
        """
        self.__scorers[scorer_id].set_tools(tools)

    def update_tools(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            est: Optional[Union[BaseEstimator, BaseMixture]] = None,
            scorer_id: Optional[int] = None,
            *args, **kwargs
            ) -> type(None):
        """
        Fit internal tools of the i-th scorer to passed
        training data and, optionally, before that replace
        these tools with a new ones based on the passed instance
        of `est`.

        :param X_train:
            feature representation of training objects
        :param y_train:
            target labels
        :param est:
            instance such that new tools of the i-th scorer
            are based on it (e.g., if the i-th scorer is instance of
            `CommitteeScorer`, committee of `est` clones fitted to
            different folds becomes its new tools)
        :param scorer_id:
            identifier (number) of scorer; if it is not passed,
            all scorers are affected
        :return:
            None
        """
        if scorer_id is not None:
            scorer_ids = [scorer_id]
        else:
            scorer_ids = range(len(self.__scorers))
        for position in scorer_ids:
            self.__scorers[position].update_tools(
                X_train, y_train, est, *args, **kwargs
            )
