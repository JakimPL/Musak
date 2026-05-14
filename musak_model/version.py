from typing import Final, TypeAlias

SchemaVersion: TypeAlias = tuple[int, int, int]

TOKENIZER_SCHEMA_VERSION: Final[SchemaVersion] = (0, 1, 0)
