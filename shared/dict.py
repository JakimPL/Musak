from typing import Any


def get_key(data: dict[str, Any], key: str) -> Any:
    element = data[key]
    if isinstance(element, list) and len(element) == 1:
        return element[0]

    return element
