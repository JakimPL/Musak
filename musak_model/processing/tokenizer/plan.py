from musak_model.processing.parser import ParsedScoreArtifact


def missing_tokenization_sources(
    parsed_scores: tuple[ParsedScoreArtifact, ...],
    *,
    completed_source_ids: set[str],
) -> tuple[ParsedScoreArtifact, ...]:
    return tuple(artifact for artifact in parsed_scores if artifact.source_id_value not in completed_source_ids)


def batched_tokenization_sources(
    parsed_scores: tuple[ParsedScoreArtifact, ...],
    *,
    batch_size: int,
) -> tuple[tuple[ParsedScoreArtifact, ...], ...]:
    return tuple(parsed_scores[start : start + batch_size] for start in range(0, len(parsed_scores), batch_size))
