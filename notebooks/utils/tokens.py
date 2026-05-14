from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, Token


def token_label(token: Token, *, duration_vocabulary: DurationVocabulary) -> str:
    return token.to_text(duration_vocabulary=duration_vocabulary)


def token_rows(tokens: list[Token], *, duration_vocabulary: DurationVocabulary) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    bar_index = 0
    position = 0
    for token in tokens:
        rows.append(
            {
                "bar": bar_index + 1,
                "position": position,
                "token": token_label(token, duration_vocabulary=duration_vocabulary),
                "kind": token.kind,
            }
        )
        position += 1
        if isinstance(token, BarToken):
            bar_index += 1
            position = 0
        elif isinstance(token, EndToken):
            break

    return rows


def default_duration_vocabulary() -> DurationVocabulary:
    return DurationVocabulary(TokenizationConfig.load())
