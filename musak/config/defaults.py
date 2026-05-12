from typing import Final

SEQUENTIAL: Final[bool] = False

MELODIC: Final[bool] = False

TEMPO: Final[int] = 120
MIN_TEMPO: Final[int] = 60
MAX_TEMPO: Final[int] = 240

GROUPS: Final[int] = 1
MIN_GROUPS: Final[int] = 1
MAX_GROUPS: Final[int] = 4

MEASURES: Final[int] = 2
MIN_MEASURES: Final[int] = 1
MAX_MEASURES: Final[int] = 8

TIME_SIGNATURE: Final[tuple[int, int]] = (4, 4)
TIME_SIGNATURE_NUMERATOR: Final[int] = 4
MIN_TIME_SIGNATURE_NUMERATOR: Final[int] = 1
MAX_TIME_SIGNATURE_NUMERATOR: Final[int] = 32

TIME_SIGNATURE_DENOMINATOR: Final[int] = 4
MIN_TIME_SIGNATURE_DENOMINATOR: Final[int] = 1
MAX_TIME_SIGNATURE_DENOMINATOR: Final[int] = 32

LOWEST_NOTE: Final[int] = 40
MIN_LOWEST_NOTE: Final[int] = 21
MAX_LOWEST_NOTE: Final[int] = 108

HIGHEST_NOTE: Final[int] = 90
MIN_HIGHEST_NOTE: Final[int] = 21
MAX_HIGHEST_NOTE: Final[int] = 108
