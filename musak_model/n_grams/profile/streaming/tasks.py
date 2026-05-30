from collections.abc import Iterable, Iterator
from fractions import Fraction
from pathlib import Path

from musak_model.n_grams.profile.streaming.schema import FigureBatchTask
from musak_model.tokens.config import TokenizationConfig
from musak_model.training.ingestion.schema import EncodedExercise


def figure_batch_tasks(
    encoded_jsonl_path: Path,
    *,
    tokenization_config: TokenizationConfig,
    min_n: int,
    max_n: int,
    rhythm_min_n: int,
    rhythm_max_n: int,
    grid_alignment_denominators: tuple[int, ...],
    strong_beat_offsets: tuple[Fraction, ...],
    register_arch_basis_count: int,
    batch_size: int,
    completed_batches: set[int],
) -> Iterator[FigureBatchTask]:
    with encoded_jsonl_path.open("r", encoding="utf-8") as file:
        batch_index = 0
        sample_start_index = 0
        encoded_lines: list[str] = []
        for line in file:
            if line.strip() == "":
                continue

            encoded_lines.append(line)
            if len(encoded_lines) == batch_size:
                if batch_index not in completed_batches:
                    yield _figure_batch_task(
                        batch_index=batch_index,
                        sample_start_index=sample_start_index,
                        encoded_lines=encoded_lines,
                        tokenization_config=tokenization_config,
                        min_n=min_n,
                        max_n=max_n,
                        rhythm_min_n=rhythm_min_n,
                        rhythm_max_n=rhythm_max_n,
                        grid_alignment_denominators=grid_alignment_denominators,
                        strong_beat_offsets=strong_beat_offsets,
                        register_arch_basis_count=register_arch_basis_count,
                    )
                batch_index += 1
                sample_start_index += len(encoded_lines)
                encoded_lines.clear()

        if encoded_lines and batch_index not in completed_batches:
            yield _figure_batch_task(
                batch_index=batch_index,
                sample_start_index=sample_start_index,
                encoded_lines=encoded_lines,
                tokenization_config=tokenization_config,
                min_n=min_n,
                max_n=max_n,
                rhythm_min_n=rhythm_min_n,
                rhythm_max_n=rhythm_max_n,
                grid_alignment_denominators=grid_alignment_denominators,
                strong_beat_offsets=strong_beat_offsets,
                register_arch_basis_count=register_arch_basis_count,
            )


def figure_sample_batch_tasks(
    samples: Iterable[EncodedExercise],
    *,
    tokenization_config: TokenizationConfig,
    min_n: int,
    max_n: int,
    rhythm_min_n: int,
    rhythm_max_n: int,
    grid_alignment_denominators: tuple[int, ...],
    strong_beat_offsets: tuple[Fraction, ...],
    register_arch_basis_count: int,
    batch_size: int,
    completed_batches: set[int],
) -> Iterator[FigureBatchTask]:
    batch_index = 0
    sample_start_index = 0
    encoded_lines: list[str] = []
    for sample in samples:
        encoded_lines.append(sample.model_dump_json())
        if len(encoded_lines) == batch_size:
            if batch_index not in completed_batches:
                yield _figure_batch_task(
                    batch_index=batch_index,
                    sample_start_index=sample_start_index,
                    encoded_lines=encoded_lines,
                    tokenization_config=tokenization_config,
                    min_n=min_n,
                    max_n=max_n,
                    rhythm_min_n=rhythm_min_n,
                    rhythm_max_n=rhythm_max_n,
                    grid_alignment_denominators=grid_alignment_denominators,
                    strong_beat_offsets=strong_beat_offsets,
                    register_arch_basis_count=register_arch_basis_count,
                )
            batch_index += 1
            sample_start_index += len(encoded_lines)
            encoded_lines.clear()

    if encoded_lines and batch_index not in completed_batches:
        yield _figure_batch_task(
            batch_index=batch_index,
            sample_start_index=sample_start_index,
            encoded_lines=encoded_lines,
            tokenization_config=tokenization_config,
            min_n=min_n,
            max_n=max_n,
            rhythm_min_n=rhythm_min_n,
            rhythm_max_n=rhythm_max_n,
            grid_alignment_denominators=grid_alignment_denominators,
            strong_beat_offsets=strong_beat_offsets,
            register_arch_basis_count=register_arch_basis_count,
        )


def _figure_batch_task(
    *,
    batch_index: int,
    sample_start_index: int,
    encoded_lines: list[str],
    tokenization_config: TokenizationConfig,
    min_n: int,
    max_n: int,
    rhythm_min_n: int,
    rhythm_max_n: int,
    grid_alignment_denominators: tuple[int, ...],
    strong_beat_offsets: tuple[Fraction, ...],
    register_arch_basis_count: int,
) -> FigureBatchTask:
    return FigureBatchTask(
        batch_index=batch_index,
        sample_start_index=sample_start_index,
        encoded_lines=tuple(encoded_lines),
        tokenization_config=tokenization_config,
        min_n=min_n,
        max_n=max_n,
        rhythm_min_n=rhythm_min_n,
        rhythm_max_n=rhythm_max_n,
        grid_alignment_denominators=grid_alignment_denominators,
        strong_beat_offsets=strong_beat_offsets,
        register_arch_basis_count=register_arch_basis_count,
    )
