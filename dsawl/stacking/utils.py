"""
A collection of auxiliaries for stacking.

@author: Nikolay Lysenko.
"""


from typing import List, Tuple, Optional, Any

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


class InitablePipeline(Pipeline):
    """
    A pipeline that can be instantiated without any arguments.
    """

    def __init__(
            self,
            steps: Optional[List[Tuple[str, Any]]] = None,
            memory: Optional[str] = None
            ):
        if steps is None:
            super().__init__(
                steps=[('arbitrary_step', LinearRegression())],
                memory=memory
            )
        else:
            super().__init__(steps, memory)  # pragma: no cover (unnecessary)
