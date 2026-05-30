from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from musak_model.synthetic.processes.accent import AccentFieldOverride
from musak_model.synthetic.processes.pitch import RegisterCurveOverride


class FittedGeneratorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    register_overrides: tuple[RegisterCurveOverride, ...] = ()
    accent_overrides: tuple[AccentFieldOverride, ...] = ()

    @classmethod
    def read(cls, path: Path) -> FittedGeneratorConfig:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
