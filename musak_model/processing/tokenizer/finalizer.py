from dataclasses import dataclass

from musak_model.processing.manifest import EncodedManifestField
from musak_model.processing.tokenizer.output import (
    EncodedManifestAppender,
    clear_tokenized_source_temp_files,
    iter_encoded_manifest_rows,
)
from musak_model.processing.tokenizer.resume import TokenizationOutputPaths
from musak_model.processing.tokenizer.schema import TokenizedSourceResult
from musak_shared.files import append_text_lines_from_index


@dataclass(frozen=True)
class TokenizationMergeCounts:
    encoded_line_count: int
    manifest_row_count: int
    encoded_count: int


def merge_tokenized_source_result(
    result: TokenizedSourceResult,
    *,
    output_paths: TokenizationOutputPaths,
    encoded_line_count: int,
    manifest_row_count: int,
    encoded_count: int,
) -> TokenizationMergeCounts:
    line_mapping = append_text_lines_from_index(
        result.temp_encoded_jsonl_path,
        output_paths.encoded_jsonl_path,
        start_line_index=encoded_line_count,
    )
    with EncodedManifestAppender(output_paths.encoded_manifest_path) as manifest_appender:
        for row in iter_encoded_manifest_rows(result.temp_encoded_manifest_path):
            global_row: dict[str, object] = dict(_global_manifest_row(row, line_mapping=line_mapping))
            manifest_appender.append(global_row)

    clear_tokenized_source_temp_files(
        encoded_jsonl_path=result.temp_encoded_jsonl_path,
        encoded_manifest_path=result.temp_encoded_manifest_path,
    )
    return TokenizationMergeCounts(
        encoded_line_count=encoded_line_count + len(line_mapping),
        manifest_row_count=manifest_row_count + result.manifest_row_count,
        encoded_count=encoded_count + result.encoded_count,
    )


def _global_manifest_row(row: dict[str, str], *, line_mapping: dict[int, int]) -> dict[str, str]:
    encoded_line = row[EncodedManifestField.ENCODED_LINE.value]
    if encoded_line == "":
        return row

    return row | {EncodedManifestField.ENCODED_LINE.value: str(line_mapping[int(encoded_line)])}
