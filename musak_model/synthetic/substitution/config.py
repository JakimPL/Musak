from pydantic import BaseModel, ConfigDict, Field


class SubstitutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lambda_curve: float = Field(ge=0.0)
    lambda_harm: float = Field(ge=0.0)
    lambda_accent: float = Field(ge=0.0)
    commonness_bias: float = Field(ge=0.0)
    max_resample_retries: int = Field(gt=0)
