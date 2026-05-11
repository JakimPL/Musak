import pathlib
from typing import TypeVar

import yaml
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml(path: pathlib.Path, model: type[ModelT]) -> ModelT:
    with open(path, encoding="utf-8") as file:
        return model.model_validate(yaml.safe_load(file))
