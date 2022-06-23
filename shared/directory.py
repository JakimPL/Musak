import os
import uuid

TEMP_DIRECTORY = 'temp'


def create_directory() -> tuple[str, str]:
    uuid64 = str(uuid.uuid4())
    if not os.path.isdir(TEMP_DIRECTORY):
        os.mkdir(TEMP_DIRECTORY)

    temp_directory = os.path.join(TEMP_DIRECTORY, uuid64)
    os.mkdir(temp_directory)
    return uuid64, temp_directory
