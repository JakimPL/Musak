from starlette.datastructures import FormData


def form_str(form: FormData, key: str, default: str | int = "") -> str:
    value = form.get(key)
    return value if isinstance(value, str) else str(default)
