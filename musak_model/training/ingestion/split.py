from pathlib import Path
from random import Random
from typing import Final

from music21.exceptions21 import Music21Exception

from musak_model.common.files import collect_musicxml_files
from musak_model.data.pipeline import process_file
from musak_model.data.schema import Segment
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration_vocabulary import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, Hand, Token
from musak_model.tokens.vocabulary import TokenVocabulary, encode
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.ingestion.schema import EncodedExercise, IngestionErrorRecord, IngestionSplit

_FILE_PROCESSING_ERRORS: Final[tuple[type[Exception], ...]] = (
    Music21Exception,
    OSError,
    TypeError,
    ValueError,
)


def build_split(
    source_dir: Path,
    *,
    config: IngestionConfig,
) -> IngestionSplit:
    """Build deterministic train/validation split from MusicXML files."""
    file_paths = collect_musicxml_files(source_dir)
    validation_files = set(
        _split_validation_files(
            file_paths=file_paths,
            validation_fraction=config.validation_fraction,
            split_seed=config.split_seed,
        )
    )

    train_samples: list[EncodedExercise] = []
    validation_samples: list[EncodedExercise] = []
    invalid_files: list[IngestionErrorRecord] = []
    duration_vocabulary = DurationVocabulary(TokenizationConfig.load())
    token_vocabulary = TokenVocabulary(duration_vocabulary)

    for file_path in file_paths:
        try:
            segments = process_file(
                file_path,
                segmentation=config.segmentation,
                difficulty_labels=config.difficulty_labels,
                duration_vocabulary=duration_vocabulary,
            )
        except _FILE_PROCESSING_ERRORS as exception:
            invalid_files.append(
                IngestionErrorRecord(
                    file=str(file_path),
                    exception_type=type(exception).__name__,
                    message=str(exception),
                )
            )
            continue

        encoded_samples = _encode_segments(segments, token_vocabulary=token_vocabulary)
        if file_path in validation_files:
            validation_samples.extend(encoded_samples)
        else:
            train_samples.extend(encoded_samples)

    return IngestionSplit(train=train_samples, validation=validation_samples, invalid_files=invalid_files)


def _split_validation_files(*, file_paths: list[Path], validation_fraction: float, split_seed: int) -> list[Path]:
    if not file_paths:
        return []

    shuffled_files = list(file_paths)
    Random(split_seed).shuffle(shuffled_files)
    validation_count = _validation_count(total_files=len(shuffled_files), validation_fraction=validation_fraction)
    return shuffled_files[:validation_count]


def _validation_count(*, total_files: int, validation_fraction: float) -> int:
    if total_files <= 1 or validation_fraction <= 0:
        return 0

    count = int(total_files * validation_fraction)
    if count == 0:
        count = 1

    if count >= total_files:
        count = total_files - 1

    return count


def _encode_segments(segments: list[Segment], *, token_vocabulary: TokenVocabulary) -> list[EncodedExercise]:
    encoded_samples: list[EncodedExercise] = []
    for segment in segments:
        encoded_samples.append(_encode_segment(segment, token_vocabulary=token_vocabulary))

    return encoded_samples


def _encode_segment(segment: Segment, *, token_vocabulary: TokenVocabulary) -> EncodedExercise:
    tokens = segment.tokens if segment.tokens else segment.right_hand_tokens + segment.left_hand_tokens
    token_ids = encode(tokens, vocabulary=token_vocabulary)
    bar_positions = _build_bar_positions_from_tokens(tokens)
    return EncodedExercise(
        token_ids=token_ids,
        bar_positions=bar_positions,
        hand=None,
        metadata=segment.metadata,
    )


def _encode_segment_hands(segment: Segment, *, token_vocabulary: TokenVocabulary) -> list[EncodedExercise]:
    right_hand = _encode_hand_segment(
        segment=segment,
        tokens=segment.right_hand_tokens,
        hand=Hand.RIGHT,
        token_vocabulary=token_vocabulary,
    )
    left_hand = _encode_hand_segment(
        segment=segment,
        tokens=segment.left_hand_tokens,
        hand=Hand.LEFT,
        token_vocabulary=token_vocabulary,
    )
    return [right_hand, left_hand]


def _encode_hand_segment(
    *,
    segment: Segment,
    tokens: list[Token],
    hand: Hand,
    token_vocabulary: TokenVocabulary,
) -> EncodedExercise:
    token_ids = encode(tokens, vocabulary=token_vocabulary)
    bar_positions = _build_bar_positions_from_tokens(tokens)
    return EncodedExercise(
        token_ids=token_ids,
        bar_positions=bar_positions,
        hand=hand,
        metadata=segment.metadata,
    )


def _build_bar_positions_from_tokens(tokens: list[Token]) -> list[int]:
    bar_index = 0
    bar_positions: list[int] = []

    for token in tokens:
        if isinstance(token, EndToken):
            bar_positions.append(max(bar_index - 1, 0))
            continue

        bar_positions.append(bar_index)
        if isinstance(token, BarToken):
            bar_index += 1

    return bar_positions
