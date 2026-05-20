from musak_model.data.schema import SegmentIneligibilityReason


class TokenizationIneligibilityError(ValueError):
    def __init__(self, message: str, *, reason: SegmentIneligibilityReason) -> None:
        super().__init__(message)
        self.reason = reason
