from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .common import sample_key, write_csv

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _column_number(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for c in letters:
        n = n * 26 + ord(c) - 64
    return n - 1


def read_xlsx(path: Path, sheet_name: str) -> List[Dict[str, str]]:
    """Dependency-free OOXML reader for scalar cells; preserves blank columns."""
    ns = {"m": MAIN_NS, "r": REL_NS}
    with zipfile.ZipFile(path) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        targets = {x.attrib["Id"]: x.attrib["Target"] for x in rels}
        shared: List[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            sx = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iter(f"{{{MAIN_NS}}}t")) for si in sx]
        sheet = next((s for s in wb.findall(".//m:sheet", ns) if s.attrib["name"] == sheet_name), None)
        if sheet is None:
            names = [s.attrib["name"] for s in wb.findall(".//m:sheet", ns)]
            raise ValueError(f"sheet {sheet_name!r} not found; available={names}")
        target = targets[sheet.attrib[f"{{{REL_NS}}}id"]].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        root = ET.fromstring(z.read(target))
        matrix: List[List[str]] = []
        for row in root.findall(".//m:sheetData/m:row", ns):
            values: Dict[int, str] = {}
            for cell in row.findall("m:c", ns):
                idx = _column_number(cell.attrib["r"])
                typ = cell.attrib.get("t")
                value_node = cell.find("m:v", ns)
                value = "" if value_node is None else (value_node.text or "")
                if typ == "s" and value:
                    value = shared[int(value)]
                elif typ == "inlineStr":
                    value = "".join(t.text or "" for t in cell.iter(f"{{{MAIN_NS}}}t"))
                values[idx] = value
            width = max(values, default=-1) + 1
            matrix.append([values.get(i, "") for i in range(width)])
    if not matrix:
        return []
    headers = matrix[0]
    return [{h: row[i] if i < len(row) else "" for i, h in enumerate(headers) if h} for row in matrix[1:]]


def build_manifest(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cols = cfg["columns"]
    source = read_xlsx(Path(cfg["paths"]["labels_file"]), cfg["paths"]["labels_sheet"])
    required = [cols[x] for x in ("video_id", "clip_id", "text", "intensity", "polarity")]
    actual = list(source[0]) if source else []
    missing_columns = [x for x in required if x not in actual]
    if missing_columns:
        raise ValueError(f"label columns missing: {missing_columns}; actual={actual}")
    rows: List[Dict[str, Any]] = []
    for i, src in enumerate(source, start=2):
        vid = str(src[cols["video_id"]]).strip()
        cid = str(src[cols["clip_id"]]).strip()
        if cid.endswith(".0") and cid[:-2].lstrip("-").isdigit():
            cid = cid[:-2]
        path = Path(cfg["paths"]["dataset_dir"]) / vid / f"{cid}.mp4"
        rows.append({
            "source_row": i, "video_id": vid, "clip_id": cid, "sample_key": sample_key(vid, cid),
            "text": src[cols["text"]], "intensity": src[cols["intensity"]],
            "polarity": src[cols["polarity"]], "video_path": str(path.resolve()),
            "video_exists": path.is_file(), "video_size_bytes": path.stat().st_size if path.is_file() else 0,
        })
    counts = Counter(r["sample_key"] for r in rows)
    discovered = {sample_key(p.parent.name, p.stem): p.resolve() for p in Path(cfg["paths"]["dataset_dir"]).glob("*/*.mp4")}
    labelled = {r["sample_key"] for r in rows}
    report = {
        "label_rows": len(rows), "expected_samples": cfg["runtime"]["expected_samples"],
        "duplicate_keys": sorted(k for k, n in counts.items() if n > 1),
        "missing_videos": [r["sample_key"] for r in rows if not r["video_exists"]],
        "unlabelled_videos": sorted(set(discovered) - labelled),
        "matched": sum(r["video_exists"] for r in rows), "label_columns": actual,
        "empty_text_keys": [r["sample_key"] for r in rows if not str(r["text"]).strip()],
        "invalid_intensity": [r["sample_key"] for r in rows if not _is_float(r["intensity"])],
        "polarity_values": sorted({str(r["polarity"]) for r in rows}),
    }
    return rows, report


def save_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ["source_row", "video_id", "clip_id", "sample_key", "text", "intensity", "polarity", "video_path", "video_exists", "video_size_bytes"]
    write_csv(path, rows, fields)


def _is_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
