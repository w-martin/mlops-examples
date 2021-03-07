from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class DecisionNode:
    feature: int = None
    value: float = None
    left: Union[int, 'DecisionNode'] = None
    right: Union[int, 'DecisionNode'] = None
    left_group: Optional[np.ndarray] = None
    right_group: Optional[np.ndarray] = None
