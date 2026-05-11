from starlette.datastructures import FormData


def form_str(form: FormData, key: str, default: str | int = "") -> str:
    """Extract a string value from FormData, ignoring UploadFile entries."""
    value = form.get(key)
    return value if isinstance(value, str) else str(default)
