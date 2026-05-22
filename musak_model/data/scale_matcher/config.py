from pydantic import BaseModel, ConfigDict, Field


class ScaleMatcherConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    support_score_margin: float = Field(ge=0, le=1)
    selection_score_margin: float = Field(ge=0, le=1)
    maximum_unexplained_weight_fraction: float = Field(ge=0, le=1)
    maximum_explanation_pitch_class_count: int = Field(ge=0, le=12)
