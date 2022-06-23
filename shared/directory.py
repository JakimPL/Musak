import os
import uuid


def create_directory() -> tuple[str, str]:
    uuid64 = str(uuid.uuid4())
    temp_directory = os.path.join('temp', uuid64)
    os.mkdir(temp_directory)
    return uuid64, temp_directory
