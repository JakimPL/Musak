from pathlib import Path
from typing import Final
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, ParseError
from zipfile import BadZipFile, ZipFile

_MXL_CONTAINER_PATH: Final[str] = "META-INF/container.xml"
_MXL_SUFFIX: Final[str] = ".mxl"
_MUSICXML_TITLE_FIELDS: Final[tuple[str, ...]] = ("movement-title", "work-title")


def score_title(path: Path) -> str:
    try:
        root = _musicxml_root(path)
    except (KeyError, OSError, ParseError, BadZipFile):
        return ""

    for field in _MUSICXML_TITLE_FIELDS:
        title = _first_musicxml_text(root, field)
        if title != "":
            return title

    return ""


def _musicxml_root(path: Path) -> Element:
    if path.suffix.lower() == _MXL_SUFFIX:
        return _compressed_musicxml_root(path)

    return ElementTree.parse(path).getroot()


def _compressed_musicxml_root(path: Path) -> Element:
    with ZipFile(path) as archive:
        rootfile_path = _mxl_rootfile_path(archive)
        with archive.open(rootfile_path) as file:
            return ElementTree.parse(file).getroot()


def _mxl_rootfile_path(archive: ZipFile) -> str:
    with archive.open(_MXL_CONTAINER_PATH) as file:
        container = ElementTree.parse(file).getroot()

    for element in container.iter():
        if _xml_local_name(element.tag) != "rootfile":
            continue

        full_path = element.attrib.get("full-path")
        if full_path is not None and full_path != "":
            return full_path

    raise KeyError("missing MusicXML rootfile in MXL container")


def _first_musicxml_text(root: Element, field_name: str) -> str:
    for element in root.iter():
        if _xml_local_name(element.tag) != field_name or element.text is None:
            continue

        title = element.text.strip()
        if title != "":
            return title

    return ""


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]
