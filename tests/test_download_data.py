"""Offline, tiny archives only. No GitHub request, data array, model, or GPU."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
spec = importlib.util.spec_from_file_location("download_data", MODULE)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def archive_bytes(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as bundle:
        for name, value, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(value) if kind == tarfile.REGTYPE else 0
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = "../../outside"
            bundle.addfile(member, io.BytesIO(value) if kind == tarfile.REGTYPE else None)
    return stream.getvalue()


def fixture(entries=None, files=None):
    files = files if files is not None else {"data/manifest.json": b'{"synthetic":true}', "data/a.txt": b"tiny"}
    entries = entries if entries is not None else [(key, value, tarfile.REGTYPE) for key, value in files.items()]
    content = archive_bytes(entries)
    shard = {"name": "data-00000.tar.gz", "size": len(content), "sha256": digest(content),
             "files": [{"path": key, "size": len(value), "sha256": digest(value)} for key, value in files.items()]}
    index = {"schema": d.SCHEMA, "bundle_id": "synthetic-v1", "format": "tar.gz", "total_files": len(files),
             "total_bytes": sum(map(len, files.values())), "shards": [shard]}
    return index, json.dumps(index).encode(), content


class FakeGitHub:
    def __init__(self, raw, content):
        self.raw, self.content, self.calls = raw, content, []

    def release_assets(self, tag):
        self.calls.append(("release", tag))
        return {"data-index.json": {"id": 1, "size": len(self.raw)},
                "data-00000.tar.gz": {"id": 2, "size": len(self.content)}}

    def asset(self, record):
        self.calls.append(("asset", record["id"]))
        return io.BytesIO(self.raw if record["id"] == 1 else self.content)


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[1] / "runs" / "download_synthetic_tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def install(self, entries=None, files=None):
        index, raw, content = fixture(entries, files)
        client = FakeGitHub(raw, content)
        result = d.download(index, raw, client, d.DEFAULT_TAG, self.root, progress=lambda _: None)
        return index, raw, content, client, result

    def test_round_trip_and_completed_shard_skip(self):
        index, raw, content, client, result = self.install()
        self.assertEqual(result["new_files"], 2)
        self.assertEqual(d.verify_install(index, self.root)["files"], 2)
        client.calls.clear()
        again = d.download(index, raw, client, d.DEFAULT_TAG, self.root, progress=lambda _: None)
        self.assertEqual(again["reused_shards"], 1)
        self.assertEqual(again["new_files"], 0)
        self.assertNotIn(("asset", 2), client.calls)
        self.assertFalse(list((self.root / ".data-cache").glob("*.partial")))

    def test_interrupted_partial_download_restarts(self):
        index, raw, content = fixture()
        cache = d.create_cache(self.root)
        (cache / "data-00000.tar.gz.partial").write_bytes(b"unfinished")
        result = d.download(index, raw, FakeGitHub(raw, content), d.DEFAULT_TAG, self.root, progress=lambda _: None)
        self.assertEqual(result["new_files"], 2)

    def test_release_index_is_commit_pinned(self):
        index, raw, content = fixture()
        with self.assertRaisesRegex(d.DownloadError, "cloned index"):
            d.download(index, raw, FakeGitHub(raw.replace(b"synthetic", b"different"), content), d.DEFAULT_TAG, self.root)
        self.assertFalse((self.root / ".data-cache").exists())

    def test_wrong_shard_hash_retains_partial_only(self):
        index, raw, content = fixture()
        altered = content[:-1] + bytes([content[-1] ^ 1])
        with self.assertRaisesRegex(d.DownloadError, "SHA256"):
            d.download(index, raw, FakeGitHub(raw, altered), d.DEFAULT_TAG, self.root)
        self.assertFalse((self.root / "data").exists())
        self.assertTrue((self.root / ".data-cache/data-00000.tar.gz.partial").is_file())
        self.assertFalse((self.root / ".data-cache/data-00000.tar.gz").exists())

    def test_short_and_overlong_streams_rejected(self):
        expected = {"size": 2, "sha256": digest(b"ab")}
        for raw in (b"a", b"abc"):
            with self.subTest(raw=raw), self.assertRaises(d.DownloadError):
                d.copy_verified(io.BytesIO(raw), io.BytesIO(), expected)

    def test_existing_unrelated_file_never_overwritten(self):
        (self.root / "data").mkdir()
        target = self.root / "data/a.txt"
        target.write_bytes(b"user work")
        with self.assertRaisesRegex(d.DownloadError, "overwrite"):
            self.install()
        self.assertEqual(target.read_bytes(), b"user work")

    def test_other_existing_data_file_is_preserved(self):
        (self.root / "data").mkdir()
        target = self.root / "data/unrelated.txt"
        target.write_bytes(b"keep")
        self.install()
        self.assertEqual(target.read_bytes(), b"keep")

    def test_bad_member_paths_rejected(self):
        for name in ("../x", "/data/x", "C:/data/x", "data/../x", "data//x", "data/./x",
                     "data/x\\z", "data/a:stream", "data/NUL.txt", "data/LPT1", "data/trailing.",
                     "data/trailing ", "data/x\n", "elsewhere/x"):
            with self.subTest(name=name), self.assertRaises(d.DownloadError):
                d.safe_member(name)

    def test_traversal_in_archive_rejected_before_install(self):
        entries = [("data/manifest.json", b'{"synthetic":true}', tarfile.REGTYPE), ("data/../evil", b"x", tarfile.REGTYPE)]
        with self.assertRaises(d.DownloadError):
            self.install(entries)
        self.assertFalse((self.root / "data").exists())
        self.assertFalse((self.root / "evil").exists())

    def test_symlink_hardlink_device_directory_entries_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.DIRTYPE):
            with self.subTest(kind=kind):
                index, raw, content = fixture([("data/a.txt", b"", kind)], {"data/a.txt": b""})
                with tempfile.TemporaryDirectory(dir=self.root) as target, self.assertRaises(d.DownloadError):
                    d.download(index, raw, FakeGitHub(raw, content), d.DEFAULT_TAG, Path(target), progress=lambda _: None)

    def test_duplicate_missing_and_unexpected_members_rejected(self):
        files = {"data/a.txt": b"a", "data/b.txt": b"b"}
        cases = [[("data/a.txt", b"a", tarfile.REGTYPE)],
                 [("data/a.txt", b"a", tarfile.REGTYPE)] * 2,
                 [("data/a.txt", b"a", tarfile.REGTYPE), ("data/c.txt", b"b", tarfile.REGTYPE)]]
        for entries in cases:
            index, raw, content = fixture(entries, files)
            with tempfile.TemporaryDirectory(dir=self.root) as target, self.assertRaises(d.DownloadError):
                d.download(index, raw, FakeGitHub(raw, content), d.DEFAULT_TAG, Path(target), progress=lambda _: None)
            self.assertFalse((self.root / "data").exists())

    def test_member_hash_is_checked_before_any_publish(self):
        entries = [("data/manifest.json", b'{"synthetic":true}', tarfile.REGTYPE), ("data/a.txt", b"evil", tarfile.REGTYPE)]
        with self.assertRaisesRegex(d.DownloadError, "SHA256"):
            self.install(entries)
        self.assertFalse((self.root / "data").exists())

    def test_index_counts_types_duplicates_and_prefix_collision(self):
        original, _, _ = fixture()
        mutations = [lambda x: x.update(total_files=True), lambda x: x.update(total_bytes=0),
                     lambda x: x["shards"][0].update(size=True),
                     lambda x: x["shards"][0]["files"][1].update(path="data/MANIFEST.json"),
                     lambda x: x["shards"][0]["files"][1].update(path="data/manifest.json/child"),
                     lambda x: x["shards"][0].update(name="../shard.tar.gz")]
        for change in mutations:
            index = copy.deepcopy(original)
            change(index)
            with self.assertRaises(d.DownloadError):
                d.validate_index(index)

    def test_json_duplicate_keys_and_nonfinite_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'not json'):
            with self.assertRaises(d.DownloadError):
                d.parse_json(raw)

    def test_cache_unowned_content_rejected(self):
        cache = self.root / ".data-cache"
        cache.mkdir()
        (cache / "user.txt").write_text("keep")
        with self.assertRaisesRegex(d.DownloadError, "ownership"):
            d.create_cache(self.root)

    def test_destination_link_rejected(self):
        target = self.root / "target"
        target.mkdir()
        link = self.root / "data"
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            self.skipTest("OS does not permit test symlink creation")
        with self.assertRaisesRegex(d.DownloadError, "Symlink"):
            self.install()

    def test_token_env_priority_and_gh_no_secret_in_argv(self):
        def forbidden(*args, **kwargs):
            self.fail("Should not invoke gh when env token exists")
        self.assertEqual(d.get_token({"GH_TOKEN": "secret1", "GITHUB_TOKEN": "secret2"}, forbidden), "secret1")
        def fake(args, **kwargs):
            self.assertEqual(args, ["gh", "auth", "token", "--hostname", "github.com"])
            self.assertTrue(kwargs["capture_output"])
            return types.SimpleNamespace(returncode=0, stdout="private-token\n", stderr="")
        self.assertEqual(d.get_token({}, fake), "private-token")

    def test_auth_failure_has_no_secret_stderr(self):
        fake = lambda *a, **kw: types.SimpleNamespace(returncode=1, stdout="secret", stderr="secret-url")
        with self.assertRaises(d.DownloadError) as caught:
            d.get_token({}, fake)
        self.assertNotIn("secret", str(caught.exception))
        with self.assertRaises(d.DownloadError):
            d.get_token({"GH_TOKEN": "token\r\nInjected: true"})

    def test_redirect_strips_token_and_rejects_untrusted_hosts(self):
        request = urllib.request.Request("https://api.github.com/repos/u/r/releases/assets/1", headers={"Authorization": "Bearer private-token"})
        request.add_unredirected_header("Authorization", "Bearer private-token")
        redirect = d.SafeRedirect()
        good = redirect.redirect_request(request, None, 302, "", {}, "https://release-assets.githubusercontent.com/file?signature=private")
        self.assertNotIn("Authorization", good.headers)
        self.assertNotIn("Authorization", good.unredirected_hdrs)
        for url in ("http://release-assets.githubusercontent.com/file", "https://evil.test/file",
                    "https://api.github.com:444/file", "https://user:pass@api.github.com/file"):
            with self.assertRaises(d.DownloadError):
                redirect.redirect_request(request, None, 302, "", {}, url)

    def test_api_ignores_remote_asset_url_and_sanitizes_http_error(self):
        seen = []
        class Opener:
            def open(inner, request, **kwargs):
                seen.append(request)
                raise urllib.error.HTTPError("https://secret-signed-url/?secret", 403, "secret", {}, None)
        github = d.GitHub("owner/repo", "private-token", Opener())
        with self.assertRaises(d.DownloadError) as caught:
            github.asset({"id": 3, "url": "https://evil.test/"})
        self.assertEqual(seen[0].full_url, "https://api.github.com/repos/owner/repo/releases/assets/3")
        self.assertNotIn("secret", str(caught.exception))
        self.assertNotIn("private-token", str(caught.exception))

    def test_release_assets_are_paginated_and_exact_tag_bound(self):
        github = d.GitHub("owner/repo", "token")
        calls = []
        def fake(path):
            calls.append(path)
            if "/tags/" in path:
                return {"tag_name": "v1", "id": 2, "draft": False}
            if path.endswith("&page=1"):
                return [{"name": str(i), "id": i + 1, "size": 1, "state": "uploaded"} for i in range(100)]
            return [{"name": "last", "id": 101, "size": 1, "state": "uploaded"}]
        github.json = fake
        self.assertEqual(len(github.release_assets("v1")), 101)
        self.assertTrue(calls[-1].endswith("page=2"))
        with self.assertRaises(d.DownloadError):
            github.release_assets("other")

    def test_verify_only_does_not_authenticate_or_network(self):
        index, raw, _, _, _ = self.install()
        index_path = self.root / "data-index.json"
        index_path.write_bytes(raw)
        with patch.object(d, "get_token", side_effect=AssertionError("auth forbidden")), \
                patch.object(d, "GitHub", side_effect=AssertionError("network forbidden")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(d.main(["--index", str(index_path), "--destination", str(self.root), "--verify-only"]), 0)

    def test_publication_never_replaces_concurrent_target(self):
        source, target = self.root / "source", self.root / "target"
        source.write_bytes(b"source")
        target.write_bytes(b"keep")
        with self.assertRaises(d.DownloadError):
            d.publish_no_replace(source, target)
        self.assertEqual(target.read_bytes(), b"keep")

    def test_failed_network_read_is_sanitized_by_cli(self):
        index, raw, _ = fixture()
        path = self.root / "data-index.json"
        path.write_bytes(raw)
        errors = io.StringIO()
        with patch.object(d, "get_token", return_value="private-token"), \
                patch.object(d, "download", side_effect=OSError("https://signed-secret/?token=private-token")), \
                contextlib.redirect_stderr(errors):
            self.assertEqual(d.main(["--index", str(path)]), 1)
        self.assertNotIn("private-token", errors.getvalue())
        self.assertNotIn("signed-secret", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
