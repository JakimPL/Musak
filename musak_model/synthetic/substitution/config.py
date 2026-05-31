from pydantic import BaseModel, ConfigDict, Field

from musak_model.synthetic.substitution.texture import ALL_MELODIC_TEXTURE, HandTextureConfig


class SubstitutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lambda_curve: float = Field(ge=0.0)
    lambda_harmonic: float = Field(ge=0.0)
    lambda_accent: float = Field(ge=0.0)
    lambda_chord_figure: float = Field(ge=0.0)
    commonness_bias: float = Field(ge=0.0)
    max_resample_retries: int = Field(gt=0)
    monophonic: bool
    texture: HandTextureConfig = ALL_MELODIC_TEXTURE
