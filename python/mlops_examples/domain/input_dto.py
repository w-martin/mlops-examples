from typing import List

from pydantic.main import BaseModel


class InputDto(BaseModel):
    data: List[List[float]]
