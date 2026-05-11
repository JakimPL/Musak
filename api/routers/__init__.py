from starlette.datastructures import FormData


def form_str(form: FormData, key: str, default: str = "") -> str:
    """Extract a string value from FormData, ignoring UploadFile entries."""
    v = form.get(key)
    return v if isinstance(v, str) else default
