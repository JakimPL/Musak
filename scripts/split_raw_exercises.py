from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

RAW_LEVEL_PATTERN: Final[re.Pattern[str]] = re.compile(r"^Level (?P<level>\d+)\.mxl$")
MXL_CONTAINER_PATH: Final[str] = "META-INF/container.xml"
DEFAULT_SCORE_PATH: Final[str] = "score.xml"
FINAL_BAR_STYLE: Final[str] = "light-heavy"
OUTPUT_NAME_WIDTH: Final[int] = 4
DEFAULT_DIFFICULTY_LABELS_NAME: Final[str] = "difficulty_labels.json"
STRIPPED_SCORE_METADATA_TAGS: Final[frozenset[str]] = frozenset(
    {
        "work",
        "movement-number",
        "movement-title",
        "identification",
        "credit",
    }
)


@dataclass(frozen=True)
class RawLevel:
    level: int
    path: Path


def main() -> None:
    arguments = _parse_arguments()
    raw_levels = _raw_levels(arguments.raw_dir)
    if not raw_levels:
        raise ValueError(f"no raw level files found in {arguments.raw_dir}")

    total_outputs = 0
    difficulty_labels: dict[str, int] = {}
    for raw_level in raw_levels:
        output_paths = split_raw_level(
            raw_level,
            output_dir=arguments.output_dir,
            force=arguments.force,
        )
        difficulty_labels.update(
            _difficulty_labels(output_paths, output_dir=arguments.output_dir, level=raw_level.level)
        )
        total_outputs += len(output_paths)
        print(f"Level {raw_level.level}: wrote {len(output_paths)} exercises")

    _write_difficulty_labels(arguments.difficulty_labels_output, difficulty_labels)
    print(f"Wrote {total_outputs} exercises into {arguments.output_dir}")
    print(f"Wrote difficulty labels to {arguments.difficulty_labels_output}")


def split_raw_level(raw_level: RawLevel, *, output_dir: Path, force: bool) -> list[Path]:
    level_output_dir = output_dir / str(raw_level.level)
    if level_output_dir.exists():
        if not force:
            raise FileExistsError(f"{level_output_dir} already exists; pass --force to replace it")
        shutil.rmtree(level_output_dir)

    with tempfile.TemporaryDirectory(prefix=f"musak-level-{raw_level.level}-") as temporary_directory:
        temporary_output_dir = Path(temporary_directory) / str(raw_level.level)
        temporary_output_dir.mkdir(parents=True)

        source_archive = _read_mxl(raw_level.path)
        score_root = ET.fromstring(source_archive.score_xml)
        exercises = _split_score(score_root)

        output_paths = []
        for index, exercise_root in enumerate(exercises, start=1):
            output_path = temporary_output_dir / f"{index:0{OUTPUT_NAME_WIDTH}d}.mxl"
            exercise_xml = _serialize_score(exercise_root)
            _write_mxl(
                output_path,
                score_xml=exercise_xml,
                container_xml=source_archive.container_xml,
                score_path=source_archive.score_path,
            )
            output_paths.append(level_output_dir / output_path.name)

        level_output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temporary_output_dir), str(level_output_dir))

    return output_paths


@dataclass(frozen=True)
class MxlArchive:
    score_path: str
    score_xml: bytes
    container_xml: bytes


def _read_mxl(path: Path) -> MxlArchive:
    with zipfile.ZipFile(path) as archive:
        container_xml = archive.read(MXL_CONTAINER_PATH)
        score_path = _score_path_from_container(container_xml)
        return MxlArchive(
            score_path=score_path,
            score_xml=archive.read(score_path),
            container_xml=container_xml,
        )


def _score_path_from_container(container_xml: bytes) -> str:
    container_root = ET.fromstring(container_xml)
    rootfile = container_root.find("./rootfiles/rootfile")
    if rootfile is None:
        return DEFAULT_SCORE_PATH

    full_path = rootfile.attrib.get("full-path")
    if full_path is None:
        return DEFAULT_SCORE_PATH

    return full_path


def _split_score(score_root: ET.Element) -> list[ET.Element]:
    parts = _parts(score_root)
    part_measures = [_measures(part) for part in parts]
    if not part_measures:
        raise ValueError("score has no parts")

    split_indices = _split_indices(part_measures[0])
    if not split_indices:
        raise ValueError("score contains no final barlines")

    for measures in part_measures[1:]:
        if len(measures) != len(part_measures[0]):
            raise ValueError("score parts do not have matching measure counts")
        if _split_indices(measures) != split_indices:
            raise ValueError("score parts do not have matching final barline positions")

    exercises = []
    start_index = 0
    for split_index in split_indices:
        end_index = split_index + 1
        exercises.append(_score_slice(score_root, start_index=start_index, end_index=end_index))
        start_index = end_index

    if start_index != len(part_measures[0]):
        raise ValueError("trailing measures found after the last final barline")

    return exercises


def _parts(score_root: ET.Element) -> list[ET.Element]:
    return [child for child in score_root if child.tag == "part"]


def _measures(part: ET.Element) -> list[ET.Element]:
    return [child for child in part if child.tag == "measure"]


def _split_indices(measures: list[ET.Element]) -> list[int]:
    return [index for index, measure in enumerate(measures) if _has_final_barline(measure)]


def _has_final_barline(measure: ET.Element) -> bool:
    for barline in measure.findall("barline"):
        bar_style = barline.find("bar-style")
        if bar_style is not None and bar_style.text == FINAL_BAR_STYLE:
            return True
    return False


def _score_slice(score_root: ET.Element, *, start_index: int, end_index: int) -> ET.Element:
    output_root = copy.deepcopy(score_root)
    _strip_score_metadata(output_root)
    source_parts = _parts(score_root)
    output_parts = _parts(output_root)

    for source_part, output_part in zip(source_parts, output_parts, strict=True):
        output_part[:] = [copy.deepcopy(measure) for measure in _measures(source_part)[start_index:end_index]]
        _carry_leading_attributes(source_part, output_part, start_index=start_index)

    return output_root


def _strip_score_metadata(score_root: ET.Element) -> None:
    score_root[:] = [child for child in score_root if child.tag not in STRIPPED_SCORE_METADATA_TAGS]


def _carry_leading_attributes(source_part: ET.Element, output_part: ET.Element, *, start_index: int) -> None:
    output_measures = _measures(output_part)
    if not output_measures:
        raise ValueError("cannot carry attributes into an empty part")

    leading_measure = output_measures[0]
    leading_attributes = leading_measure.find("attributes")
    if leading_attributes is None:
        leading_attributes = ET.Element("attributes")
        leading_measure.insert(_attributes_insert_index(leading_measure), leading_attributes)

    _merge_carried_attributes(leading_attributes, _active_attributes(source_part, before_measure_index=start_index))


def _attributes_insert_index(measure: ET.Element) -> int:
    for index, child in enumerate(measure):
        if child.tag != "print":
            return index
    return len(measure)


def _merge_carried_attributes(leading_attributes: ET.Element, carried: list[ET.Element]) -> None:
    existing_children = list(leading_attributes)
    merged_children: list[ET.Element] = []
    consumed_ids: set[int] = set()

    for tag in ("divisions", "key", "time", "staves"):
        existing = leading_attributes.find(tag)
        if existing is not None:
            merged_children.append(existing)
            consumed_ids.add(id(existing))
            continue

        carried_attribute = _first_attribute(carried, tag)
        if carried_attribute is not None:
            merged_children.append(copy.deepcopy(carried_attribute))

    existing_clefs = leading_attributes.findall("clef")
    if existing_clefs:
        merged_children.extend(existing_clefs)
        consumed_ids.update(id(clef) for clef in existing_clefs)
    else:
        merged_children.extend(copy.deepcopy(attribute) for attribute in carried if attribute.tag == "clef")

    merged_children.extend(child for child in existing_children if id(child) not in consumed_ids)
    leading_attributes[:] = merged_children


def _first_attribute(attributes: list[ET.Element], tag: str) -> ET.Element | None:
    for attribute in attributes:
        if attribute.tag == tag:
            return attribute
    return None


def _active_attributes(source_part: ET.Element, *, before_measure_index: int) -> list[ET.Element]:
    active_by_tag: dict[str, ET.Element] = {}
    active_clefs: dict[str, ET.Element] = {}
    for measure in _measures(source_part)[: before_measure_index + 1]:
        attributes = measure.find("attributes")
        if attributes is None:
            continue
        for child in attributes:
            if child.tag == "clef":
                active_clefs[child.attrib.get("number", "")] = child
            else:
                active_by_tag[child.tag] = child

    active_attributes = [active_by_tag[tag] for tag in ("divisions", "key", "time", "staves") if tag in active_by_tag]
    active_attributes.extend(active_clefs[key] for key in sorted(active_clefs))
    return active_attributes


def _serialize_score(score_root: ET.Element) -> bytes:
    xml_body = ET.tostring(score_root, encoding="unicode")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
        f"{xml_body}\n"
    ).encode("utf-8")


def _write_mxl(output_path: Path, *, score_xml: bytes, container_xml: bytes, score_path: str) -> None:
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MXL_CONTAINER_PATH, container_xml)
        archive.writestr(score_path, score_xml)


def _difficulty_labels(output_paths: list[Path], *, output_dir: Path, level: int) -> dict[str, int]:
    return {output_path.relative_to(output_dir).as_posix(): level for output_path in output_paths}


def _write_difficulty_labels(path: Path, labels: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split raw Level N.mxl exercise collections into standalone MXL files."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/exercises/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/exercises"))
    parser.add_argument(
        "--difficulty-labels-output",
        type=Path,
        default=Path("data/exercises") / DEFAULT_DIFFICULTY_LABELS_NAME,
        help="JSON path-to-difficulty mapping to write.",
    )
    parser.add_argument("--force", action="store_true", help="replace existing per-level output directories")
    return parser.parse_args()


def _raw_levels(raw_dir: Path) -> list[RawLevel]:
    levels = []
    for path in raw_dir.iterdir():
        match = RAW_LEVEL_PATTERN.match(path.name)
        if match is None:
            continue
        levels.append(RawLevel(level=int(match.group("level")), path=path))

    return sorted(levels, key=lambda raw_level: raw_level.level)


if __name__ == "__main__":
    main()
