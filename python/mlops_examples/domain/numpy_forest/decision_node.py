from dataclasses import dataclass
from typing import Union


@dataclass
class DecisionNode:
    feature: int = None
    value: float = None
    left: Union[int, 'DecisionNode'] = None
    right: Union[int, 'DecisionNode'] = None
