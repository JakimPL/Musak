from typing import Any, Protocol


class HasBaseNoteName(Protocol):
    def _asdict(self) -> dict[str, Any]: ...
    def get_base_note_name(self) -> str: ...


def namedtuple_with_base_note(obj: HasBaseNoteName) -> dict[str, Any]:
    data: dict[str, Any] = dict(obj._asdict())
    data["base_note"] = obj.get_base_note_name()
    return data


def get_key(data: dict[str, Any], key: str) -> Any:
    element = data.get(key)
    if isinstance(element, list) and len(element) == 1:
        return element[0]

    return element
