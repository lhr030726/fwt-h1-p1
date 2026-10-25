"""Freeze metadata-only RCDA spatial groups and bounded development subsets.

This tool never loads NPY arrays, infers label semantics, or certifies historical
non-exposure. A separately reviewed, hash-bound source audit is mandatory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from .artifacts import atomic_json, output_path, read_json, sha256
from .manifest import MANIFEST_SCHEMA, ROLES, catalog_files, manifest_digest, validate_manifest

AUDIT_SCHEMA = "fwt_rcda_source_review_v1"
SPLIT_SCHEMA = "fwt_rcda_spatial_split_v1"
DEVELOPMENT_ROLES = ROLES[:-1]
WEIGHTS = (55, 15, 10, 10, 10)
SANITY_EVENT_COUNTS = dict(backbone_train=2, residual_train=2, validation=4)
_TEXT_EVIDENCE_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".csv", ".py", ".js", ".wkt", ".prj"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def catalog_digest(catalog):
    """Bind sorted identities and exact input/label paths, without reading arrays."""
    rows = sorted(catalog["samples"], key=lambda row: (row["sample_id"], row["source_split"]))
    return _digest(dict(data_root=str(Path(catalog["data_root"]).resolve()), samples=rows,
                        excluded=catalog.get("excluded", [])))


def _rank(seed, purpose, identity):
    return _digest([seed, purpose, identity]), identity


def _review_source(catalog, audit, affine_path, crs_path):
    if not isinstance(audit, dict) or audit.get("schema_version") != AUDIT_SCHEMA:
        raise ValueError("A separate source audit with schema " + AUDIT_SCHEMA + " is required")
    required = dict(label_semantics="next_day_cumulative_burn", dates_verified=True,
                    geometry_verified=True)
    if any(audit.get(key) != value for key, value in required.items()):
        raise ValueError("Source audit must verify cumulative next-day labels and source geometry")
    for key in ("label_date_evidence", "exposure_review", "reviewed_by"):
        if not isinstance(audit.get(key), str) or not audit[key].strip():
            raise ValueError("Source audit requires " + key)
    try:
        reviewed_at = datetime.fromisoformat(audit["reviewed_at"])
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("Source audit requires an actual timezone-aware reviewed_at") from exc
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise ValueError("reviewed_at must include a timezone")
    if Path(audit.get("data_root", "")).resolve() != Path(catalog["data_root"]).resolve():
        raise ValueError("Source audit data_root differs from the catalog")
    if audit.get("catalog_sha256") != catalog_digest(catalog):
        raise ValueError("Source audit catalog_sha256 differs from the current filename catalog")
    if audit.get("affine_audit_sha256") != sha256(affine_path):
        raise ValueError("Source audit affine_audit_sha256 mismatch")
    if audit.get("crs_sha256") != sha256(crs_path):
        raise ValueError("Source audit crs_sha256 mismatch")
    evidence = audit.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Source audit requires nonempty hash-bound evidence")
    for item in evidence:
        if not isinstance(item, dict) or not isinstance(item.get("description"), str) or not item["description"].strip():
            raise ValueError("Each source evidence item requires a description")
        path = Path(item.get("path", "")).resolve()
        if path.suffix.lower() not in _TEXT_EVIDENCE_SUFFIXES:
            raise ValueError("Only text/source evidence may be read by the split preparer")
        if not path.is_file() or item.get("sha256") != sha256(path):
            raise ValueError("Source evidence missing or SHA256 mismatch: " + str(path))


def footprint_groups(catalog, affine_records):
    """Connect all years by any positive-area intersection of 256x256 footprints.

    Bounds use pixel outer edges. Edge-only contact is not an intersection.
    The supported source geometry is explicitly north-up 50 m, with no shear.
    """
    events = sorted({row["event_id"] for row in catalog["samples"]})
    transforms = {}
    for item in affine_records:
        eid = f"{item['year']}_UID_{int(item['uid'])}"
        if eid not in events:
            continue
        transform = item.get("transform")
        if item.get("valid_6tuple") is not True or not isinstance(transform, list) or len(transform) != 6:
            raise ValueError("Invalid affine metadata for " + eid)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in transform):
            raise ValueError("Nonfinite affine metadata for " + eid)
        if transform[1:3] != [50, 0] or transform[4:] != [0, -50]:
            raise ValueError("Expected north-up 50 m source grid for " + eid)
        if eid in transforms:
            raise ValueError("Duplicate affine metadata for " + eid)
        transforms[eid] = transform
    missing = sorted(set(events) - set(transforms))
    if missing:
        raise ValueError("Missing affine metadata: " + ", ".join(missing[:5]))
    parent = {eid: eid for eid in events}

    def find(eid):
        while parent[eid] != eid:
            parent[eid] = parent[parent[eid]]
            eid = parent[eid]
        return eid

    boxes = {eid: (a[0], a[3] - 12800, a[0] + 12800, a[3]) for eid, a in transforms.items()}
    edges = []
    for i, eid in enumerate(events):
        a = boxes[eid]
        for other in events[i + 1:]:
            b = boxes[other]
            if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                edges.append([eid, other])
                parent[find(other)] = find(eid)
    components = defaultdict(list)
    for eid in events:
        components[find(eid)].append(eid)
    groups = {"spatial_" + _digest(members)[:24]: members for members in components.values()}
    return groups, transforms, edges


def _hamilton(n):
    counts = [n * weight // 100 for weight in WEIGHTS]
    order = sorted(range(len(WEIGHTS)), key=lambda i: (-(n * WEIGHTS[i] % 100), i))
    for i in order[:n - sum(counts)]:
        counts[i] += 1
    return counts


def _counts(rows):
    result = {}
    for role in ROLES:
        subset = [row for row in rows if row["role"] == role]
        result[role] = dict(samples=len(subset), events=len({row["event_id"] for row in subset}),
                            groups=len({row["group_id"] for row in subset}),
                            samples_by_year=dict(sorted(Counter(row["date"][:4] for row in subset).items())),
                            events_by_year=dict(sorted(Counter(eid[:4] for eid in {row["event_id"] for row in subset}).items())))
    return result


def prepare_manifests(data_root, source_audit, affine_audit, crs_file, *, seed=0):
    """Return full/sanity/report documents after validating audit evidence.

    Does not write outputs, load arrays, select by labels, fit normalization,
    launch training, or evaluate calibration/final roles.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    affine_path, crs_path = Path(affine_audit).resolve(), Path(crs_file).resolve()
    if affine_path.suffix.lower() != ".json" or crs_path.suffix.lower() not in {".prj", ".wkt"}:
        raise ValueError("Geometry inputs must be reviewed JSON and text WKT/PRJ metadata")
    catalog = catalog_files(data_root)
    if catalog.get("excluded"):
        raise ValueError("Catalog contains excluded/missing files; resolve their metadata before freezing")
    audit = read_json(source_audit) if isinstance(source_audit, (str, Path)) else copy.deepcopy(source_audit)
    _review_source(catalog, audit, affine_path, crs_path)
    crs = crs_path.read_text(encoding="utf-8-sig").strip()
    if not crs.startswith(("PROJCS[", "PROJCRS[")):
        raise ValueError("crs_file must contain a projected CRS WKT")
    groups, transforms, edges = footprint_groups(catalog, read_json(affine_path))
    legacy_test = {row["event_id"] for row in catalog["samples"] if row["source_split"] == "test"}
    assigned, strata = {}, defaultdict(list)
    for gid, members in groups.items():
        if legacy_test.intersection(members):
            assigned[gid] = "final"
        else:
            strata[min(eid[:4] for eid in members)].append(gid)
    stratum_allocations = {}
    for year, members in sorted(strata.items()):
        ordered = sorted(members, key=lambda gid: _rank(seed, "full_split:" + year, gid))
        counts, start = _hamilton(len(ordered)), 0
        for role, count in zip(DEVELOPMENT_ROLES, counts):
            for gid in ordered[start:start + count]:
                assigned[gid] = role
            start += count
        stratum_allocations[year] = dict(zip(DEVELOPMENT_ROLES, counts))
    event_group = {eid: gid for gid, members in groups.items() for eid in members}
    rows = []
    for source in sorted(catalog["samples"], key=lambda row: (row["event_id"], row["date"])):
        row = copy.deepcopy(source)
        gid = event_group[row["event_id"]]
        row.update(role=assigned[gid], group_id=gid,
                   target_date_status="source_reviewed_next_day",
                   georeference=dict(crs=crs, affine_transform=transforms[row["event_id"]], shape=[256, 256]))
        rows.append(row)
    frozen_audit = dict(label_semantics=audit["label_semantics"], dates_verified=True, roles_frozen=True,
                        exposure_review=audit["exposure_review"], source_review=audit,
                        source_review_sha256=_digest(audit), source_reviewed_at=audit["reviewed_at"])
    split_protocol = dict(schema_version=SPLIT_SCHEMA, seed=seed,
                          grouping="all_year_positive_area_footprint_connected_components",
                          test_rule="any_legacy_test_event_reserves_entire_component_for_final",
                          stratification="component_earliest_year", allocation="Hamilton_largest_remainder_by_component_count",
                          weights=dict(zip(DEVELOPMENT_ROLES, WEIGHTS)),
                          ordering="sha256_json_seed_purpose_identity; lexical_identity_tiebreak",
                          stratum_allocations=stratum_allocations, arrays_read=False,
                          geometry_evidence=dict(affine_audit_path=str(affine_path), affine_audit_sha256=sha256(affine_path),
                                                 crs_path=str(crs_path), crs_sha256=sha256(crs_path)))
    full = validate_manifest(dict(schema_version=MANIFEST_SCHEMA, data_root=str(Path(data_root).resolve()),
                                  audit=frozen_audit, split_protocol=split_protocol, samples=rows), require_all_roles=True)
    selected = {}
    for role, n_events in SANITY_EVENT_COUNTS.items():
        gids = sorted((gid for gid, value in assigned.items() if value == role),
                      key=lambda gid: _rank(seed, "sanity_group:" + role, gid))
        if len(gids) < n_events:
            raise ValueError(f"Need {n_events} distinct groups for sanity role {role}; got {len(gids)}")
        selected[role] = [min(groups[gid], key=lambda eid: _rank(seed, "sanity_event:" + role, eid))
                          for gid in gids[:n_events]]
    selected_ids = set()
    for role, events in selected.items():
        for eid in events:
            candidates = [row for row in full["samples"] if row["event_id"] == eid]
            selected_ids.update(row["sample_id"] for row in sorted(candidates, key=lambda row: (row["date"], row["sample_id"]))[:2])
    sanity = copy.deepcopy(full)
    sanity["samples"] = [row for row in sanity["samples"] if row["role"] not in SANITY_EVENT_COUNTS or row["sample_id"] in selected_ids]
    sanity["sanity_subset"] = dict(purpose="engineering_check_only_not_formal_evidence", parent_manifest_sha256=manifest_digest(full),
                                    event_counts=SANITY_EVENT_COUNTS, max_dates_per_event=2,
                                    selection="distinct_groups_hash_ranked; lowest_event_hash; earliest_two_dates",
                                    selected_events=selected, retained_metadata_only_roles=list(ROLES[3:]),
                                    permitted_array_roles=list(SANITY_EVENT_COUNTS), arrays_read=False)
    sanity = validate_manifest(sanity, require_all_roles=True)
    report = dict(schema_version=SPLIT_SCHEMA, catalog_sha256=catalog_digest(catalog),
                  full_manifest_sha256=manifest_digest(full), sanity_manifest_sha256=manifest_digest(sanity),
                  full_counts=_counts(full["samples"]), sanity_counts=_counts(sanity["samples"]),
                  events=len(event_group), group_count=len(groups), overlap_edges=len(edges), max_group_events=max(map(len, groups.values())),
                  group_size_histogram=dict(sorted(Counter(map(len, groups.values())).items())),
                  original_test_events=len(legacy_test), moved_legacy_train_events=sorted({row["event_id"] for row in rows if row["role"] == "final"} - legacy_test),
                  groups={gid: dict(events=members, role=assigned[gid]) for gid, members in sorted(groups.items())},
                  overlap_edges_event_ids=edges,
                  cross_role_overlap_edges=sum(assigned[event_group[a]] != assigned[event_group[b]] for a, b in edges),
                  arrays_read=False,
                  caveats=["Legacy labels and statistics were historically audited; this does not establish never-seen final events.",
                           "Geographic footprint isolation does not establish independent fires or exchangeability.",
                           "Ratios apply to component counts within earliest-year strata, not to event/sample counts.",
                           "Calibration and final rows retained in sanity manifest are metadata only; their arrays must remain unopened."])
    return full, sanity, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-audit", required=True)
    parser.add_argument("--affine-audit", required=True)
    parser.add_argument("--crs-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    destination = output_path(args.output_dir)
    if destination.exists():
        raise ValueError("Output directory must be new; frozen split artifacts are never overwritten")
    full, sanity, report = prepare_manifests(args.data_root, args.source_audit, args.affine_audit, args.crs_file, seed=args.seed)
    for name, value in (("full_manifest.json", full), ("sanity_manifest.json", sanity), ("split_report.json", report)):
        atomic_json(destination / name, value)
    print(json.dumps(dict(output=str(destination), full=report["full_counts"], sanity=report["sanity_counts"], arrays_read=False)))


if __name__ == "__main__":
    main()
