"""Explicit, event-disjoint RCDA roles. Cataloguing never opens an array.

A catalogue is NOT a trainable manifest or evidence of historical non-exposure.
No automatic random split, legacy train/test rewrite, or label inspection occurs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

from .artifacts import atomic_json, output_path

MANIFEST_SCHEMA = "fwt_rcda_manifest_v2"
ROLES = ("backbone_train", "residual_train", "validation", "prob_calibration", "risk_calibration", "final")
_NAME = re.compile(r"^UID_FIRE_(\d+)_(\d{4}-\d{2}-\d{2})\.npy$", re.IGNORECASE)


def parse_sample_name(filename: str) -> dict:
    """Read identity only; a same-name label does not establish its target date."""
    match = _NAME.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"unrecognised RCDA sample filename: {filename}")
    uid, day = match.groups()
    current = date.fromisoformat(day)
    event_id = f"{current.year}_UID_{int(uid)}"
    return dict(sample_id=f"{event_id}/{day}", event_id=event_id, date=day,
                target_date=(current + timedelta(days=1)).isoformat())


def validate_manifest(manifest: dict, *, base_dir=None, require_all_roles: bool = False,
                      check_paths: bool = True) -> dict:
    """Return a copy with absolute paths; reject alias paths and event leakage.

    Dates are declarations checked for next-day consistency, not a certification
    that the contents or historical label exposure have been independently audited.
    Existing file identities additionally catch hard-link aliases without reading.
    """
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"schema_version must be {MANIFEST_SCHEMA}")
    result = copy.deepcopy(manifest)
    rows = result.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest requires a nonempty samples list")
    base = Path(base_dir or ".").resolve()
    data_root = Path(result.get("data_root", "."))
    if not data_root.is_absolute():
        data_root = base / data_root
    ids, paths, file_ids, event_dates, event_roles = set(), set(), set(), set(), {}
    present_roles = set()
    group_roles = {}
    fields = ("sample_id", "event_id", "date", "target_date", "input_path", "label_path", "role")
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key] for key in fields):
            raise ValueError(f"each sample requires nonempty strings: {fields}")
        role, sid, eid = row["role"], row["sample_id"], row["event_id"]
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; roles must be explicitly assigned")
        try:
            day, target_day = date.fromisoformat(row["date"]), date.fromisoformat(row["target_date"])
        except ValueError as exc:
            raise ValueError(f"invalid date for {sid}") from exc
        if target_day - day != timedelta(days=1):
            raise ValueError(f"sample {sid} is not an exact next-day pair")
        if not re.fullmatch(rf"{day.year}_UID_\d+", eid):
            raise ValueError(f"event_id must include year and UID: {eid}")
        if sid in ids:
            raise ValueError(f"duplicate sample_id: {sid}")
        if (eid, row["date"]) in event_dates:
            raise ValueError(f"duplicate event/date identity: {eid}, {row['date']}")
        ids.add(sid)
        event_dates.add((eid, row["date"]))
        if eid in event_roles and event_roles[eid] != role:
            raise ValueError(f"event {eid} crosses role boundaries: {event_roles[eid]} and {role}")
        event_roles[eid] = role
        group = row.get('group_id', eid)
        if not isinstance(group, str) or not group:
            raise ValueError('group_id must be a nonempty string')
        if group in group_roles and group_roles[group] != role:
            raise ValueError(f'spatial group {group} crosses role boundaries')
        group_roles[group] = role
        present_roles.add(role)
        for field in ("input_path", "label_path"):
            path = Path(row[field])
            path = (data_root / path if not path.is_absolute() else path).resolve()
            canonical = os.path.normcase(str(path))
            if canonical in paths:
                raise ValueError(f"duplicate path alias: {path}")
            paths.add(canonical)
            if check_paths:
                if not path.is_file():
                    raise ValueError(f"missing sample path: {path}")
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if stat.st_ino and identity in file_ids:
                    raise ValueError(f"duplicate file path identity: {path}")
                if stat.st_ino:
                    file_ids.add(identity)
            parsed_match = _NAME.fullmatch(path.name)
            if field == "input_path" and parsed_match:
                identity = parse_sample_name(path.name)
                if identity["event_id"] != eid or identity["date"] != row["date"]:
                    raise ValueError(f"sample metadata disagrees with filename identity: {path}")
            row[field] = str(path)
    if require_all_roles and present_roles != set(ROLES):
        raise ValueError(f"missing required roles: {sorted(set(ROLES) - present_roles)}")
    result["data_root"] = str(data_root.resolve())
    return result


def load_manifest(path, *, require_all_roles=False, check_paths=True) -> dict:
    source = Path(path).resolve()
    return validate_manifest(json.loads(source.read_text(encoding="utf-8-sig")), base_dir=source.parent,
                             require_all_roles=require_all_roles, check_paths=check_paths)


def samples_for_role(manifest: dict, role: str) -> list:
    if role not in ROLES:
        raise ValueError(f"unknown role: {role}")
    rows = [row for row in manifest["samples"] if row["role"] == role]
    if not rows:
        raise ValueError(f"manifest has no samples for role: {role}")
    return rows


def manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def catalog_files(data_root) -> dict:
    """List conventional train/test input names and matching labels, never np.load."""
    root = Path(data_root).resolve()
    rows, excluded = [], []
    for legacy_split in ("train", "test"):
        inputs, labels = root / legacy_split / "inputs", root / legacy_split / "labels"
        if not inputs.is_dir() or not labels.is_dir():
            raise ValueError(f"missing conventional inputs/labels directories under {root / legacy_split}")
        for source in sorted(inputs.glob("*.npy")):
            try:
                row = parse_sample_name(source.name)
            except ValueError:
                excluded.append(dict(input_path=str(source), reason="unrecognised_filename"))
                continue
            label = labels / source.name
            if not label.is_file():
                excluded.append(dict(input_path=str(source), reason="missing_label_filename"))
                continue
            row.update(input_path=str(source), label_path=str(label), source_split=legacy_split,
                       target_date_status="inferred_next_day_requires_source_audit", role="unassigned")
            rows.append(row)
    return dict(schema_version="fwt_rcda_filename_catalog_v1", data_root=str(root), samples=rows,
                excluded=excluded, array_contents_read=False,
                warning="Filename metadata only. Not an executable manifest, frozen split, or non-exposure audit.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    catalog = subs.add_parser("catalog", help="List filenames only; does not assign roles or read arrays")
    catalog.add_argument("--data-root", required=True)
    catalog.add_argument("--output", required=True)
    check = subs.add_parser("check", help="Check explicit role/date/path metadata, never array contents")
    check.add_argument("--manifest", required=True)
    check.add_argument("--require-all-roles", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "catalog":
        root, output = Path(args.data_root).resolve(), output_path(args.output)
        if output == root or root in output.parents:
            parser.error("catalog output must be outside the read-only data root")
        result = catalog_files(root)
        atomic_json(output, result)
        print(json.dumps(dict(output=str(output), samples=len(result["samples"]), arrays_read=False)))
    else:
        result = load_manifest(args.manifest, require_all_roles=args.require_all_roles)
        print(json.dumps(dict(samples=len(result["samples"]), sha256=manifest_digest(result), arrays_read=False)))


if __name__ == "__main__":
    main()
