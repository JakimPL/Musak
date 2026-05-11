from typing import Any, Literal, Optional

from pydantic import BaseModel


class FieldSchema(BaseModel):
    name: str
    type: Literal["integer", "boolean", "text"]
    label: str
    default: Any
    min: Optional[int] = None
    max: Optional[int] = None


class FieldGroupSchema(BaseModel):
    label: str
    fields: list[FieldSchema]


class ConfigResponse(BaseModel):
    groups: list[FieldGroupSchema]
    definitions: dict[str, Any] = {}
