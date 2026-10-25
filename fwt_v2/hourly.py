"""Audited, input-day POWER hourly wind used only as learned covariates.

The source is a patch-centre series at 50 m height, not 50 m spatial resolution
or measured gusts. The daily control and hourly summaries share these 24 UTC
hours. Reanalysis supports retrospective prediction, not a real-time claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np


HOURLY_MANIFEST_SCHEMA = "fwt_power_hourly_manifest_v1"
DERIVED_MANIFEST_SCHEMA = "fwt_power_hourly_manifest_v2_derived_provenance"
EXTRACTION_LEDGER_SCHEMA = "fwt_hourly_day_extraction_ledger_v1"
EXACT_DAY_DERIVATION = "exact_hour_key_selection_without_numeric_transformation"
ALLOWED_ROLES = ("backbone_train", "residual_train", "validation")
EXTRA_CHANNEL_NAMES = ("hourly_speed_std_m_s", "max_hourly_mean_speed_m_s",
                       "hourly_vector_coherence", "mean_adjacent_hourly_vector_change_m_s")


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def sample_identity_digest(row):
    return canonical_digest({key: row[key] for key in
                             ("sample_id", "event_id", "date", "target_date", "role", "georeference")})


def hourly_statistics(speed, direction_from_north_cw_degrees):
    """Population summaries of 24 hourly means; adjacent changes use 23 pairs.

    Direction is meteorological FROM, clockwise from geographic north.
    Near-zero mean-vector cancellation uses floating-point tolerance, not a
    physical weak-wind threshold. Calm hours contribute zero vectors.
    """
    speed = np.asarray(speed, dtype=np.float64)
    direction = np.asarray(direction_from_north_cw_degrees, dtype=np.float64)
    if speed.shape != (24,) or direction.shape != (24,):
        raise ValueError("hourly wind requires exactly 24 speed and direction values")
    if (not np.isfinite(speed).all() or not np.isfinite(direction).all()
            or (speed < 0).any() or (direction < 0).any() or (direction > 360).any()):
        raise ValueError("hourly wind contains missing or invalid speed/direction")
    radians = np.deg2rad(direction)
    u, v = -speed * np.sin(radians), -speed * np.cos(radians)
    mean_u, mean_v, mean_speed = float(u.mean()), float(v.mean()), float(speed.mean())
    length = float(np.hypot(mean_u, mean_v))
    # Opposite exact directions can leave a ~1e-16 trigonometric residual.
    if length <= np.finfo(np.float64).eps * 16 * float(speed.max()):
        mean_u = mean_v = length = 0.0
    coherence = min(1.0, length / mean_speed) if mean_speed > 0 else 0.0
    extras = (float(speed.std(ddof=0)), float(speed.max()), coherence,
              float(np.hypot(np.diff(u), np.diff(v)).mean()))
    return dict(mean_speed=mean_speed, mean_u=mean_u, mean_v=mean_v,
                mean_angle=float(np.arctan2(mean_v, mean_u)), direction_present=length > 0,
                extras=extras, u=u, v=v)


@dataclass(frozen=True)
class HourlyWind:
    manifest_sha256: str
    identity_sha256: str
    georeference_sha256: str
    mean_speed: float
    mean_angle: float
    direction_present: bool
    extras: tuple


class ExternalWindStore:
    """Verify immutable source records before opening any input or label arrays."""

    def __init__(self, path, expected_sha256, manifest):
        path = Path(path).resolve()
        content = path.read_bytes()
        self.sha256 = hashlib.sha256(content).hexdigest()
        if self.sha256 != expected_sha256:
            raise ValueError("external wind manifest SHA256 mismatch")
        document = json.loads(content.decode("utf-8-sig"))
        if document.get("schema_version") == "fwt_power_portable_manifest_v1":
            from .portable import PortableSources
            from dataclasses import replace
            sources = PortableSources(path, document)

            class RelocatedStore(ExternalWindStore):
                @staticmethod
                def _resolved(source, base):
                    return sources.resolve(source, base)

                @staticmethod
                def _source_parent(source, base):
                    return sources.logical_parent(source, base)

            original = document['original_manifest']
            original_path = sources.resolve(original['original_path'], path.parent)
            original_schema = json.loads(original_path.read_text(encoding='utf-8-sig')).get('schema_version')
            if original_schema not in (HOURLY_MANIFEST_SCHEMA, DERIVED_MANIFEST_SCHEMA):
                raise ValueError('Portable source must retain the original v1/v2 POWER schema')
            store = RelocatedStore(original_path,
                                   original['sha256'], manifest)
            self.records = {key: replace(value, manifest_sha256=self.sha256)
                            for key, value in store.records.items()}
            return
        schema = document.get("schema_version")
        if (schema not in (HOURLY_MANIFEST_SCHEMA, DERIVED_MANIFEST_SCHEMA)
                or document.get("time_standard") != "UTC"
                or document.get("height_above_surface_m") != 50
                or document.get("hours_per_sample") != 24):
            raise ValueError("external wind requires the audited 24-hour UTC POWER50M schema")
        expected_rows = {row["sample_id"]: row for row in manifest["samples"]
                         if row["role"] in ALLOWED_ROLES}
        rows = document.get("samples")
        if not isinstance(rows, list) or not rows:
            raise ValueError("external wind manifest requires sample records")
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError("duplicate external wind sample identity")
        if {row["sample_id"] for row in rows} != set(expected_rows):
            raise ValueError("external wind sample IDs must exactly cover allowed manifest roles")
        provenance = self._verify_derived_manifest(document, path.parent) if schema == DERIVED_MANIFEST_SCHEMA else None
        self.records = {}
        for record in rows:
            row = expected_rows[record["sample_id"]]
            if record.get("role") not in ALLOWED_ROLES or record.get("role") != row["role"]:
                raise ValueError("external wind role mismatch or forbidden role")
            for key in ("event_id", "date", "target_date"):
                if record.get(key) != row[key]:
                    raise ValueError(f"external wind {key} mismatch")
            if date.fromisoformat(row["target_date"]) - date.fromisoformat(row["date"]) != timedelta(days=1):
                raise ValueError("external wind target must be next day")
            identity = sample_identity_digest(row)
            geohash = canonical_digest(row["georeference"])
            if record.get("identity_sha256") != identity or record.get("georeference_sha256") != geohash:
                raise ValueError("external wind identity/georeference SHA256 mismatch")
            expected_hours = [row["date"].replace("-", "") + f"{hour:02d}" for hour in range(24)]
            if record.get("hour_keys") != expected_hours:
                raise ValueError("external wind hours must be ordered input-day UTC 00..23, never target day")
            if record.get("complete_hours") != 24 or record.get("missing_values") != 0:
                raise ValueError("external wind has missing hours or values")
            self._verify_coordinates(record, row["georeference"])
            self._verify_response(record, path.parent, expected_hours, provenance=provenance)
            stats = hourly_statistics(record["speed_m_s"], record["direction_from_north_cw_degrees"])
            for key, calculated in (("u_east_m_s", stats["u"]), ("v_north_m_s", stats["v"])):
                stored = np.asarray(record.get(key), dtype=np.float64)
                if stored.shape != (24,) or not np.allclose(stored, calculated, rtol=1e-12, atol=1e-12):
                    raise ValueError("external wind stored u/v disagree with meteorological FROM convention")
            self.records[row["sample_id"]] = HourlyWind(
                self.sha256, identity, geohash, stats["mean_speed"], stats["mean_angle"],
                stats["direction_present"], stats["extras"])

    @staticmethod
    def _verify_coordinates(record, georeference):
        from pyproj import Transformer

        a = np.asarray(georeference["affine_transform"], dtype=np.float64)
        height, width = georeference["shape"]
        x = a[0] + width / 2 * a[1] + height / 2 * a[2]
        y = a[3] + width / 2 * a[4] + height / 2 * a[5]
        lon, lat = Transformer.from_crs(georeference["crs"], "EPSG:4326", always_xy=True).transform(x, y, errcheck=True)
        stored = np.asarray([record.get("longitude"), record.get("latitude")], dtype=np.float64)
        if not np.isfinite(stored).all() or not np.allclose(stored, [lon, lat], rtol=0, atol=1e-8):
            raise ValueError("external wind coordinates must equal projected patch centre")

    @classmethod
    def _read_bound_json(cls, source, expected_sha256, base, cache=None):
        source = cls._resolved(source, base)
        key = (str(source), expected_sha256)
        if cache is not None and key in cache:
            return cache[key]
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError("external wind response SHA256 mismatch")
        response = json.loads(content.decode("utf-8-sig"))
        if cache is not None:
            cache[key] = response
        return response

    @classmethod
    def _verify_derived_manifest(cls, document, base):
        """Bind the extraction ledger and preserved raw rows before using any day."""
        ledger = cls._read_bound_json(document["extraction_ledger_path"],
                                      document["extraction_ledger_sha256"], base)
        request_plan = cls._read_bound_json(document["request_plan_path"], document["request_plan_sha256"], base)
        acquisition = cls._verify_source_plan(document, request_plan, base)
        extraction_sha = request_plan.get("extraction_script_sha256", request_plan.get("acquisition_script_sha256"))
        rows = document["samples"]
        if any(r.get("source_kind") not in ("RAW_HTTP", "DERIVED") for r in rows):
            raise ValueError("v2 external wind requires explicit RAW_HTTP or DERIVED provenance")
        derived = {r["sample_id"]: r for r in rows if r["source_kind"] == "DERIVED"}
        reused = {r["sample_id"]: r for r in rows if r["source_kind"] == "RAW_HTTP"}
        requested_groups = request_plan.get("request_groups", [])
        requested = {g["request_group_id"]: g for g in requested_groups}
        planned_ids = [sid for group in requested_groups for sid in group["sample_ids"]]
        if (len(requested) != len(requested_groups) or len(set(planned_ids)) != len(planned_ids)
                or set(planned_ids) != set(derived)
                or extraction_sha != document.get("source_script_sha256")):
            raise ValueError("external wind request plan coverage/script binding mismatch")
        entries = ledger.get("entries", [])
        if (ledger.get("schema_version") != EXTRACTION_LEDGER_SCHEMA
                or not isinstance(entries, list)
                or len({e["sample_id"] for e in entries}) != len(entries)
                or {e["sample_id"] for e in entries} != set(derived)
                or ledger.get("derived_day_count") != len(derived)
                or document.get("derived_day_records") != len(derived)
                or document.get("reused_raw_records") != len(reused)):
            raise ValueError("external wind extraction ledger coverage/count mismatch")
        if (ledger.get("request_plan_sha256") != document.get("request_plan_sha256")
                or ledger.get("extraction_script_sha256") != document.get("source_script_sha256")):
            raise ValueError("external wind extraction plan/script binding mismatch")
        groups = {}
        for entry in entries:
            record = derived[entry["sample_id"]]
            trace = {k: v for k, v in entry.items() if k not in
                     ("sample_id", "derived_response_path", "derived_response_sha256")}
            if (trace != record.get("provenance")
                    or entry.get("derived_response_path") != record["source_response_path"]
                    or entry.get("derived_response_sha256") != record["source_response_sha256"]):
                raise ValueError("external wind derived record differs from extraction ledger")
            if (trace.get("kind") != "DERIVED" or trace.get("derivation") != EXACT_DAY_DERIVATION
                    or trace.get("request_plan_sha256") != document.get("request_plan_sha256")
                    or trace.get("extraction_script_sha256") != document.get("source_script_sha256")):
                raise ValueError("external wind derivation must be an exact bound hour selection")
            group_id = trace.get("request_group_id")
            group = requested.get(group_id, {})
            if (record["sample_id"] not in group.get("sample_ids", []) or group.get("url") != record.get("request_url")
                    or trace.get("parent_start") != group.get("start") or trace.get("parent_end") != group.get("end")):
                raise ValueError("external wind parent provenance differs from frozen request group")
            parent_identity = tuple(trace.get(k) for k in ("parent_response_path", "parent_response_sha256", "parent_start", "parent_end"))
            if group_id in groups and groups[group_id] != parent_identity:
                raise ValueError("external wind request group maps to inconsistent parent responses")
            groups[group_id] = parent_identity
            if acquisition is not None:
                cls._verify_parent_receipt(trace, group, acquisition, base)
        if None in groups or ledger.get("parent_count") != len(groups) or document.get("source_parent_groups") != len(groups):
            raise ValueError("external wind parent group count mismatch")
        original = cls._read_bound_json(document["reused_source_manifest_path"],
                                       document["reused_source_manifest_sha256"], base)
        original_rows = original.get("samples", [])
        if (original.get("schema_version") != HOURLY_MANIFEST_SCHEMA
                or len({r["sample_id"] for r in original_rows}) != len(original_rows)
                or {r["sample_id"] for r in original_rows} != set(reused)):
            raise ValueError("external wind reused source must preserve the complete original v1 set")
        original_base = cls._source_parent(document["reused_source_manifest_path"], base)
        for original_row in original_rows:
            record = reused[original_row["sample_id"]]
            if ({k: v for k, v in record.items() if k not in ("source_kind", "provenance")} != original_row
                    or record.get("provenance") != dict(kind="REUSED_RAW_HTTP",
                        original_manifest_path=document["reused_source_manifest_path"],
                        original_manifest_sha256=document["reused_source_manifest_sha256"])):
                raise ValueError("external wind reused row differs from immutable original v1")
        return dict(parent_cache={}, original_base=original_base, request_groups=requested)

    @classmethod
    def _verify_source_plan(cls, document, plan, base):
        """Keep acquisition, extra recovery and extraction identities distinct.

        Legacy v2 used a single script for both acquisition and extraction.
        A separately frozen finalizer must preserve the original request groups
        and prove each additional request with its own consumed-budget receipts.
        """
        if "extraction_script_sha256" not in plan:
            if document.get("source_script_role") == "finalization/extraction":
                raise ValueError("separate extraction requires an explicit source-plan script binding")
            return None
        if (document.get("source_script_role") != "finalization/extraction"
                or document.get("extraction_script_sha256") != plan["extraction_script_sha256"]
                or document.get("source_script_sha256") != plan["extraction_script_sha256"]):
            raise ValueError("separate acquisition/extraction script identity differs")
        prior = cls._read_bound_json(plan["prior_acquisition_plan_path"], plan["prior_acquisition_plan_sha256"], base)
        recovery = cls._read_bound_json(plan["recovery_plan_path"], plan["recovery_plan_sha256"], base)
        # Coverage and all request mechanics remain the original frozen values.
        protected = ("request_groups", "acquisition_script_sha256", "full_manifest_path", "full_manifest_sha256",
                     "short_hourly_manifest_path", "short_hourly_manifest_sha256", "role_counts", "allowed_samples",
                     "cached_samples", "missing_samples", "unique_groups", "missing_hours", "transport")
        if any(plan.get(key) != prior.get(key) for key in protected):
            raise ValueError("separate extraction changed original acquisition plan coverage/identity")
        if (recovery.get("schema_version") != "fwt_full_weather_separate_recovery_plan_v1"
                or recovery.get("original_request_plan_sha256") != plan["prior_acquisition_plan_sha256"]
                or cls._resolved(recovery["original_request_plan_path"], base) != cls._resolved(plan["prior_acquisition_plan_path"], base)
                or recovery.get("original_acquisition_script_sha256") != prior.get("acquisition_script_sha256")
                or recovery.get("recovery_script_sha256") != plan.get("recovery_script_sha256")):
            raise ValueError("separate recovery original plan/script binding differs")
        scripts = ((plan["extraction_script_path"], plan["extraction_script_sha256"]),
                   (plan["recovery_script_path"], plan["recovery_script_sha256"]),
                   (recovery["original_acquisition_script_path"], recovery["original_acquisition_script_sha256"]))
        for path, digest in scripts:
            path = cls._resolved(path, base)
            if path.suffix != ".py" or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("external wind acquisition/recovery/extraction script bytes changed")
        groups = {g["request_group_id"]: g for g in prior["request_groups"]}
        failed = recovery.get("failed_groups", [])
        failed_ids = [item["group"]["request_group_id"] for item in failed]
        if (not failed or len(set(failed_ids)) != len(failed_ids)
                or recovery.get("original_groups") != len(groups)
                or recovery.get("original_successful_groups") != len(groups) - len(failed)
                or recovery.get("new_http_attempt_budget") != len(failed)
                or recovery.get("maximum_new_attempts_per_group") != 1
                or recovery.get("maximum_concurrent") != 2 or recovery.get("timeout_wall_seconds") != 40
                or recovery.get("tls_verify") is not True or recovery.get("trust_env") is not False
                or recovery.get("allow_redirects") is not False):
            raise ValueError("separate recovery finite budget/transport differs")
        for item in failed:
            group = item["group"]
            receipts = item.get("original_failed_receipts", [])
            if group != groups.get(group["request_group_id"]) or len(receipts) != 2:
                raise ValueError("separate recovery must be an exact original failed group")
            if {r.get("original_attempt") for r in receipts} != {1, 2}:
                raise ValueError("separate recovery requires two consumed original attempts")
            for ref in receipts:
                receipt = cls._read_bound_json(ref["path"], ref["sha256"], base)
                if (ref.get("success") is not False or receipt.get("success") is not False
                        or receipt.get("attempt") != ref["original_attempt"]
                        or receipt.get("request_group_id") != group["request_group_id"]
                        or receipt.get("url") != group["url"]):
                    raise ValueError("separate recovery original failure receipt differs")
        summary = cls._read_bound_json(document["recovery_summary_path"], document["recovery_summary_sha256"], base)
        results = summary.get("results", [])
        if (summary.get("recovery_plan_sha256") != plan["recovery_plan_sha256"]
                or summary.get("recovery_script_sha256") != plan["recovery_script_sha256"]
                or summary.get("new_actual_attempts") != len(failed)
                or summary.get("successful_groups") != sum(r.get("success") is True for r in results)
                or set(summary.get("failed_groups", [])) != {r["request_group_id"] for r in results if r.get("success") is not True}
                or summary.get("original_receipts_modified") is not False
                or summary.get("raw_rcda_arrays_read") != 0
                or len(results) != len(failed) or {r["request_group_id"] for r in results} != set(failed_ids)):
            raise ValueError("separate recovery completion coverage differs")
        if "transport_recovery_plan_path" in plan:
            return cls._verify_transport_recovery(document, plan, recovery, summary, groups, base)
        if summary["failed_groups"] or summary["successful_groups"] != len(failed):
            raise ValueError("separate recovery did not complete all original failed groups")
        return dict(plan_sha256=plan["recovery_plan_sha256"], failed_ids=set(failed_ids), receipt_cache={},
                    results={r["request_group_id"]: r for r in results})

    @classmethod
    def _verify_transport_recovery(cls, document, plan, recovery, failed_summary, groups, base):
        """Verify a finite second transport stage without rewriting stage one's failure."""
        for prefix in ("recovery_summary", "transport_recovery_summary"):
            if (document.get(prefix + "_sha256") != plan.get(prefix + "_sha256")
                    or cls._resolved(document[prefix + "_path"], base) != cls._resolved(plan[prefix + "_path"], base)):
                raise ValueError("transport recovery summary bindings differ")
        transport = cls._read_bound_json(plan["transport_recovery_plan_path"], plan["transport_recovery_plan_sha256"], base)
        summary = cls._read_bound_json(plan["transport_recovery_summary_path"], plan["transport_recovery_summary_sha256"], base)
        bindings = {"primary_plan": (plan["prior_acquisition_plan_path"], plan["prior_acquisition_plan_sha256"]),
                    "previous_recovery_plan": (plan["recovery_plan_path"], plan["recovery_plan_sha256"]),
                    "previous_recovery_summary": (plan["recovery_summary_path"], plan["recovery_summary_sha256"]),
                    "primary_acquisition_script": (recovery["original_acquisition_script_path"], recovery["original_acquisition_script_sha256"]),
                    "recovery_script": (plan["transport_recovery_script_path"], plan["transport_recovery_script_sha256"])}
        for name, (path, digest) in bindings.items():
            ref = transport.get(name, {})
            if ref.get("sha256") != digest or cls._resolved(ref["path"], base) != cls._resolved(path, base):
                raise ValueError("transport recovery prior plan/script binding differs")
        script = cls._resolved(plan["transport_recovery_script_path"], base)
        if script.suffix != ".py" or hashlib.sha256(script.read_bytes()).hexdigest() != plan["transport_recovery_script_sha256"]:
            raise ValueError("transport recovery script bytes changed")
        # Executable identity is historical metadata. Never execute or rehash an
        # outside-workspace curl binary during scientific data consumption.
        version = transport.get("curl_version")
        if version:
            cls._read_bound_json(version["path"], version["sha256"], base)
        group = transport.get("request_group", {})
        gid = group.get("request_group_id")
        if (transport.get("schema_version") != "fwt_full_weather_curl_transport_recovery_plan_v1"
                or group != groups.get(gid) or failed_summary["failed_groups"] != [gid]
                or failed_summary["successful_groups"] != 0 or len(recovery["failed_groups"]) != 1):
            raise ValueError("transport recovery must cover the exact failed first-recovery group")
        expected = dict(maximum_new_http_attempts=1, maximum_concurrent=1, wall_timeout_seconds=40,
                        curl_max_time_seconds=40, connect_timeout_seconds=15, tls_verify=True,
                        tls_backend="Schannel", proxy_disabled=True, curlrc_disabled=True, allow_redirects=False,
                        retries=0, raw_rcda_arrays_read=0)
        if any(transport.get(key) != value for key, value in expected.items()):
            raise ValueError("transport recovery fixed budget/TLS controls differ")
        before = recovery["original_started_attempts"] + failed_summary["new_actual_attempts"]
        if (transport.get("original_started_attempts") != recovery["original_started_attempts"]
                or transport.get("previous_separate_started_attempts") != failed_summary["new_actual_attempts"]
                or transport.get("total_started_before_this_plan") != before
                or transport.get("next_global_attempt_ordinal") != before + 1
                or plan.get("total_additional_actual_attempts") != failed_summary["new_actual_attempts"] + 1
                or plan.get("cumulative_actual_attempts") != before + 1):
            raise ValueError("transport recovery cumulative attempt accounting differs")
        stages = []
        for stage, prefix, outcome in (("separate_recovery", "recovery", "failed"),
                                       ("separate_transport_recovery2", "transport_recovery", "succeeded")):
            stages.append(dict(stage=stage, plan_path=plan[prefix + "_plan_path"], plan_sha256=plan[prefix + "_plan_sha256"],
                               script_path=plan[prefix + "_script_path"], script_sha256=plan[prefix + "_script_sha256"],
                               summary_path=plan[prefix + "_summary_path"], summary_sha256=plan[prefix + "_summary_sha256"],
                               actual_new_attempts=1, outcome=outcome))
        if plan.get("recovery_stages") != stages:
            raise ValueError("transport recovery ordered stages hide or alter earlier failure")
        refs = transport.get("prior_failed_receipts", [])
        original_refs = recovery["failed_groups"][0]["original_failed_receipts"]
        originals = {(str(cls._resolved(r["path"], base)), r["sha256"]) for r in original_refs}
        actual_refs = {(str(cls._resolved(r["path"], base)), r["sha256"]) for r in refs}
        if len(refs) != 3 or len(actual_refs) != 3 or not originals < actual_refs:
            raise ValueError("transport recovery must retain all three prior failed receipts")
        latest = None
        for ref in refs:
            receipt = cls._read_bound_json(ref["path"], ref["sha256"], base)
            if receipt.get("success") is not False or receipt.get("request_group_id") != gid or receipt.get("url") != group["url"]:
                raise ValueError("transport recovery prior failed receipt differs")
            if (str(cls._resolved(ref["path"], base)), ref["sha256"]) not in originals:
                latest = receipt
        if (latest != failed_summary["results"][0] or latest.get("execution_stage") != "separate_recovery"
                or latest.get("recovery_plan_sha256") != plan["recovery_plan_sha256"]
                or latest.get("recovery_attempt") != 1 or latest.get("original_attempts_consumed") != 2):
            raise ValueError("transport recovery omitted or relabeled failed first recovery")
        results = summary.get("results", [])
        if (summary.get("recovery_plan_sha256") != plan["transport_recovery_plan_sha256"]
                or summary.get("recovery_script_sha256") != plan["transport_recovery_script_sha256"]
                or summary.get("previous_actual_attempts") != before or summary.get("new_actual_attempts") != 1
                or summary.get("cumulative_actual_attempts") != before + 1 or summary.get("successful_groups") != 1
                or summary.get("failed_groups") != [] or summary.get("previous_receipts_modified") is not False
                or summary.get("raw_rcda_arrays_read") != 0 or len(results) != 1 or results[0].get("request_group_id") != gid):
            raise ValueError("transport recovery successful completion accounting differs")
        return dict(plan_sha256=plan["transport_recovery_plan_sha256"], failed_ids={gid}, receipt_cache={},
                    results={gid: results[0]}, transport_previous_attempts=before)

    @staticmethod
    def _resolved(path, base):
        path = Path(path)
        return (path if path.is_absolute() else base / path).resolve()

    @classmethod
    def _source_parent(cls, path, base):
        return cls._resolved(path, base).parent

    @classmethod
    def _verify_parent_receipt(cls, trace, group, acquisition, base):
        receipt = cls._read_bound_json(trace["parent_receipt_path"], trace["parent_receipt_sha256"], base,
                                      acquisition["receipt_cache"])
        gid = group["request_group_id"]
        transport = trace.get("parent_acquisition_kind") == "separate_transport_recovery2"
        if (receipt.get("success") is not True
                or receipt.get("request_group_id") != gid or receipt.get("url") != group["url"]
                or receipt.get("response_sha256") != trace["parent_response_sha256"]
                or cls._resolved(receipt["response_path"], base) != cls._resolved(trace["parent_response_path"], base)
                or receipt.get("tls_verify") is not True
                or receipt.get("allow_redirects") is not False):
            raise ValueError("external wind parent HTTP receipt binding differs")
        if transport:
            details = receipt.get("transport", {})
            if ("transport_previous_attempts" not in acquisition or gid not in acquisition["failed_ids"]
                    or receipt != acquisition["results"][gid] or receipt.get("execution_stage") != "separate_transport_recovery2"
                    or receipt.get("recovery_plan_sha256") != acquisition["plan_sha256"]
                    or receipt.get("transport_attempt") != 1 or receipt.get("prior_actual_attempts") != acquisition["transport_previous_attempts"]
                    or receipt.get("global_attempt_ordinal") != acquisition["transport_previous_attempts"] + 1
                    or receipt.get("tls_backend") != "Schannel" or receipt.get("proxy_disabled") is not True
                    or receipt.get("retries") != 0 or receipt.get("returncode") != 0
                    or details.get("http_code") != 200 or details.get("ssl_verify_result") != 0
                    or details.get("num_redirects") != 0 or details.get("num_retries") != 0
                    or details.get("proxy_used") != 0 or details.get("url_effective") != group["url"]):
                raise ValueError("external wind second transport recovery receipt differs")
            headers = cls._resolved(receipt["headers_path"], base)
            if headers.suffix != ".txt" or hashlib.sha256(headers.read_bytes()).hexdigest() != receipt.get("headers_sha256"):
                raise ValueError("external wind transport HTTP header evidence differs")
            return
        if receipt.get("status_code") != 200 or receipt.get("trust_env") is not False:
            raise ValueError("external wind primary/recovery HTTP transport differs")
        if gid in acquisition["failed_ids"]:
            if (trace.get("parent_acquisition_kind") != "separate_recovery"
                    or receipt.get("execution_stage") != "separate_recovery"
                    or receipt.get("recovery_attempt") != 1 or receipt.get("original_attempts_consumed") != 2
                    or receipt.get("recovery_plan_sha256") != acquisition["plan_sha256"]
                    or receipt != acquisition["results"][gid]):
                raise ValueError("external wind separate recovery parent receipt differs")
        elif (trace.get("parent_acquisition_kind") != "primary" or receipt.get("attempt") not in (1, 2)
              or "execution_stage" in receipt):
            raise ValueError("external wind primary acquisition receipt differs")

    @classmethod
    def _verify_response(cls, record, base, expected_hours, provenance=None):
        source_base = provenance["original_base"] if provenance is not None and record["source_kind"] == "RAW_HTTP" else base
        response = cls._read_bound_json(record["source_response_path"], record.get("source_response_sha256"), source_base)
        # Archived source requests may wrap the exact response as JSON text.
        if "response" in response:
            response = response["response"]
            if isinstance(response, str):
                response = json.loads(response)
        if provenance is None and ("derivation" in response or record.get("source_kind") == "DERIVED"):
            raise ValueError("derived wind records require the explicit v2 provenance schema")
        if provenance is not None and record["source_kind"] == "DERIVED":
            cls._verify_parent_response(record, response, base, expected_hours, provenance)
        elif "derivation" in response:
            raise ValueError("RAW_HTTP wind response cannot contain derived provenance")
        header = response.get("header", {})
        if header.get("time_standard") != "UTC":
            raise ValueError("external wind response must declare UTC")
        parameters = response.get("properties", {}).get("parameter", {})
        for parameter, field in (("WS50M", "speed_m_s"), ("WD50M", "direction_from_north_cw_degrees")):
            values = parameters.get(parameter, {})
            if set(values) != set(expected_hours):
                raise ValueError("external wind response must contain exactly the 24 input-day hours")
            if not np.array_equal(np.asarray([values[key] for key in expected_hours]), np.asarray(record[field])):
                raise ValueError("external wind values differ from source response")
        units = response.get("parameters", {})
        if units.get("WS50M", {}).get("units") != "m/s" or units.get("WD50M", {}).get("units") != "Degrees":
            raise ValueError("external wind source units must be m/s and Degrees")
        coordinates = np.asarray(response.get("geometry", {}).get("coordinates", [])[:2], dtype=np.float64)
        if coordinates.shape != (2,) or not np.allclose(
                coordinates, [record["longitude"], record["latitude"]], rtol=0, atol=.001):
            raise ValueError("external wind response coordinates differ from requested patch centre")

    @classmethod
    def _verify_parent_response(cls, record, response, base, expected_hours, provenance):
        trace = response.get("derivation")
        if trace != record["provenance"] or trace.get("exact_hour_keys") != expected_hours:
            raise ValueError("external wind day derivation differs from record/exact input-day hours")
        parent = cls._read_bound_json(trace["parent_response_path"], trace["parent_response_sha256"], base, provenance["parent_cache"])
        if "response" in parent or "derivation" in parent:
            raise ValueError("derived wind parent must be an immutable raw HTTP response, not another derived record")
        start, end = date.fromisoformat(trace["parent_start"]), date.fromisoformat(trace["parent_end"])
        selected = date.fromisoformat(record["date"])
        if not start <= selected <= end or not 0 <= (end - start).days < 31:
            raise ValueError("external wind selected day lies outside parent request range")
        header, day_header = parent.get("header", {}), response.get("header", {})
        if (header.get("time_standard") != "UTC" or header.get("start") != start.strftime("%Y%m%d")
                or header.get("end") != end.strftime("%Y%m%d") or set(header.get("sources", [])) != {"MERRA2", "POWER"}
                or day_header.get("start") != selected.strftime("%Y%m%d") or day_header.get("end") != selected.strftime("%Y%m%d")):
            raise ValueError("external wind parent/day UTC range or source metadata mismatch")
        if ({k: v for k, v in day_header.items() if k not in ("title", "start", "end")} !=
                {k: v for k, v in header.items() if k not in ("title", "start", "end")}
                or response.get("geometry") != parent.get("geometry")
                or response.get("parameters") != parent.get("parameters")):
            raise ValueError("external wind derived day metadata differs from parent response")
        url = urlsplit(record["request_url"])
        query = parse_qs(url.query)
        if (url.scheme != "https" or url.hostname != "power.larc.nasa.gov" or url.path != "/api/temporal/hourly/point"
                or len(query.get("parameters", [])) != 1 or set(query["parameters"][0].split(",")) != {"WS50M", "WD50M"}):
            raise ValueError("external wind parent request must identify the POWER hourly wind endpoint")
        required = {"start": [start.strftime("%Y%m%d")], "end": [end.strftime("%Y%m%d")], "time-standard": ["UTC"]}
        if any(query.get(k) != v for k, v in required.items()):
            raise ValueError("external wind parent request URL range/time differs from provenance")
        for key in ("longitude", "latitude"):
            values = query.get(key, [])
            if len(values) != 1 or not np.isfinite(float(values[0])) or abs(float(values[0]) - record[key]) > 1e-8:
                raise ValueError("external wind parent request URL coordinates mismatch")
        all_hours = [(start + timedelta(days=offset)).strftime("%Y%m%d") + f"{hour:02d}"
                     for offset in range((end - start).days + 1) for hour in range(24)]
        group = provenance["request_groups"][trace["request_group_id"]]
        if group.get("exact_hour_keys") != all_hours:
            raise ValueError("external wind complete parent hours differ from frozen request group")
        for key in ("longitude", "latitude"):
            value = group.get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or abs(value - record[key]) > 1e-8:
                raise ValueError("external wind frozen request group coordinates mismatch")
        parent_parameters = parent.get("properties", {}).get("parameter", {})
        day_parameters = response.get("properties", {}).get("parameter", {})
        for name, maximum in (("WS50M", None), ("WD50M", 360)):
            parent_values, day_values = parent_parameters.get(name, {}), day_parameters.get(name, {})
            if set(parent_values) != set(all_hours) or set(day_values) != set(expected_hours):
                raise ValueError("external wind parent range/day hour keys are incomplete or unexpected")
            values = np.asarray([parent_values[key] for key in all_hours], dtype=np.float64)
            if (not np.isfinite(values).all() or (values < 0).any()
                    or (maximum is not None and (values > maximum).any())
                    or (values == header.get("fill_value", -999)).any()):
                raise ValueError("external wind parent response contains missing or invalid values")
            if any(day_values[key] != parent_values[key] for key in expected_hours):
                raise ValueError("external wind derived day values differ from immutable parent response")

    def for_sample(self, row):
        if row.get("role") not in ALLOWED_ROLES:
            raise PermissionError("external wind is restricted to backbone_train/residual_train/validation")
        record = self.records.get(row["sample_id"])
        if record is None or record.identity_sha256 != sample_identity_digest(row):
            raise ValueError("external wind sample identity changed after source validation")
        return record
