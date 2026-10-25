"""Download commit-pinned release data anonymously; never start training.

Uses only Python's standard library. Completed shards are reusable; an interrupted
shard is downloaded afresh. Only --authenticated reads existing GitHub credentials;
they never enter a URL, subprocess argument, file, or diagnostic. Archive
installation is confined to regular files in data/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request

SCHEMA = "fwt_server_data_index_v1"
DEFAULT_REPO = "lhr030726/fwt-h1-p1"
DEFAULT_TAG = "rcda-h1-data-v1"
INDEX_ASSET = "data-index.json"
CHUNK = 1024 * 1024
MAX_INDEX = 64 * CHUNK
MAX_API = 8 * CHUNK
CACHE_MARKER = ".fwt-download-cache-v1"
CDN_HOSTS = {"release-assets.githubusercontent.com", "objects.githubusercontent.com"}


class DownloadError(RuntimeError):
    """An intentionally sanitized error suitable for printing."""


def require(condition, message):
    if not condition:
        raise DownloadError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            result.update(block)
    return result.hexdigest()


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def safe_member(name):
    require(isinstance(name, str) and name.startswith("data/"), "Member must be under data/.")
    require(not any(ord(c) < 32 or c in '\\:<>|?*' for c in name), "Unsafe archive member path.")
    parts = name.split("/")
    require(all(p not in ("", ".", "..") and not p.endswith((".", " ")) for p in parts),
            "Unsafe archive member components.")
    require(not PurePosixPath(name).is_absolute(), "Absolute member path forbidden.")
    reserved = {"con", "prn", "aux", "nul", "conin$", "conout$"}
    reserved |= {f"{p}{n}" for p in ("com", "lpt") for n in range(1, 10)}
    require(all(p.split(".")[0].casefold() not in reserved for p in parts), "Reserved member name.")
    return name


def unique_json(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key.")
        result[key] = value
    return result


def parse_json(raw):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_json,
                          parse_constant=lambda _: (_ for _ in ()).throw(DownloadError("Nonfinite JSON.")))
    except (UnicodeError, ValueError) as exc:
        raise DownloadError("Invalid JSON data.") from None


def validate_index(index):
    require(isinstance(index, dict) and index.get("schema") == SCHEMA, "Unknown data index schema.")
    require(isinstance(index.get("bundle_id"), str) and bool(index["bundle_id"].strip()), "Missing bundle identity.")
    require(index.get("format") == "tar.gz", "Only the pinned tar.gz format is supported.")
    shards = index.get("shards")
    require(isinstance(shards, list) and 0 < len(shards) <= 1000, "Invalid shard list.")
    seen_names, seen_paths, bytes_total = set(), set(), 0
    for shard in shards:
        require(isinstance(shard, dict), "Invalid shard record.")
        name = shard.get("name")
        require(isinstance(name, str) and re.fullmatch(r"data-[0-9]{5}\.tar\.gz", name), "Invalid shard name.")
        require(name not in seen_names, "Duplicate shard name.")
        seen_names.add(name)
        require(integer(shard.get("size"), 1) and shard["size"] < 2 * 1024**3, "Invalid shard size.")
        require(isinstance(shard.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", shard["sha256"]),
                "Invalid shard SHA256.")
        require(isinstance(shard.get("files"), list) and bool(shard["files"]), "Empty shard file list.")
        for item in shard["files"]:
            require(isinstance(item, dict), "Invalid file record.")
            path = safe_member(item.get("path"))
            folded = path.casefold()
            require(folded not in seen_paths, "Duplicate or case-aliased member.")
            seen_paths.add(folded)
            require(integer(item.get("size")), "Invalid file size.")
            require(isinstance(item.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
                    "Invalid file SHA256.")
            bytes_total += item["size"]
    for path in seen_paths:
        require(not any(str(p) in seen_paths for p in PurePosixPath(path).parents), "File/directory path collision.")
    require(integer(index.get("total_files"), 1) and index["total_files"] == len(seen_paths), "File total mismatch.")
    require(integer(index.get("total_bytes")) and index["total_bytes"] == bytes_total, "Byte total mismatch.")
    return index


def load_index(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_INDEX + 1)
    require(len(raw) <= MAX_INDEX, "Index exceeds metadata size limit.")
    return validate_index(parse_json(raw)), raw


def plain_path(path):
    """Reject symlink/junction ancestors before resolving a destination."""
    path = Path(os.path.abspath(path))
    for part in (*reversed(path.parents), path):
        try:
            metadata = part.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(metadata.st_mode) and not
                (getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)),
                "Symlink or reparse-point destination forbidden.")
    return path


def exact_file(path, spec):
    path = plain_path(path)
    if not path.exists():
        return False
    require(path.is_file(), "Expected regular file at an installation/cache path.")
    require(path.stat().st_size == spec["size"] and sha256(path) == spec["sha256"],
            "Existing file differs from the pinned data; refusing to overwrite it.")
    return True


def create_cache(root):
    cache = plain_path(root / ".data-cache")
    if cache.exists():
        require(cache.is_dir(), "Cache path is not a directory.")
        marker = plain_path(cache / CACHE_MARKER)
        if marker.exists():
            require(marker.is_file() and marker.read_bytes() == b"fwt-data-cache-v1\n", "Unknown cache ownership marker.")
        else:
            require(not any(cache.iterdir()), "Nonempty cache without ownership marker.")
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / CACHE_MARKER
    if not marker.exists():
        with marker.open("xb") as stream:
            stream.write(b"fwt-data-cache-v1\n")
    return cache


def get_token(environ=None, run=None):
    environ = os.environ if environ is None else environ
    run = subprocess.run if run is None else run
    token = environ.get("GH_TOKEN") or environ.get("GITHUB_TOKEN")
    if not token:
        try:
            result = run(["gh", "auth", "token", "--hostname", "github.com"],
                         capture_output=True, text=True, check=False, timeout=30)
        except (OSError, subprocess.SubprocessError):
            raise DownloadError("Login required: run gh auth login, or provide GH_TOKEN/GITHUB_TOKEN securely.") from None
        require(result.returncode == 0, "GitHub login unavailable; run gh auth login.")
        token = result.stdout.strip()
    require(isinstance(token, str) and bool(token) and not any(c.isspace() for c in token), "Invalid GitHub token format.")
    return token


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        require(parsed.scheme == "https" and parsed.hostname in CDN_HOSTS | {"api.github.com"}
                and parsed.port in (None, 443) and parsed.username is None and parsed.password is None,
                "Unexpected asset redirect rejected.")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            # Never forward the private token, even to a subsequent API redirect.
            redirected.remove_header("Authorization")
        return redirected


class GitHub:
    def __init__(self, repo, token=None, opener=None):
        require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) is not None, "Expected owner/repository.")
        self.repo, self._token = repo, token
        self.opener = opener or urllib.request.build_opener(SafeRedirect())

    def open(self, suffix, binary=False):
        request = urllib.request.Request("https://api.github.com/repos/" + self.repo + suffix,
                                         headers={"Accept": "application/octet-stream" if binary else "application/vnd.github+json",
                                                  "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "fwt-h1-p1-data/1"})
        if self._token:
            request.add_unredirected_header("Authorization", "Bearer " + self._token)
        try:
            return self.opener.open(request, timeout=120)
        except urllib.error.HTTPError as exc:
            if not self._token and exc.code in (403, 429):
                raise DownloadError(
                    f"Anonymous GitHub request returned HTTP {exc.code}; possible rate limit or access restriction. "
                    "Wait before retrying, or explicitly use --authenticated after gh auth login "
                    "or secure GH_TOKEN/GITHUB_TOKEN setup. No automatic retry.") from None
            raise DownloadError(f"GitHub returned HTTP {exc.code}; verify repository access and release assets.") from None
        except (urllib.error.URLError, OSError, ValueError):
            raise DownloadError("GitHub request failed; no credentials or signed URLs are included in this diagnostic.") from None

    def json(self, suffix):
        with self.open(suffix) as stream:
            raw = stream.read(MAX_API + 1)
        require(len(raw) <= MAX_API, "API metadata exceeds limit.")
        return parse_json(raw)

    def release_assets(self, tag):
        require(isinstance(tag, str) and 0 < len(tag) < 200 and not any(ord(c) < 32 for c in tag), "Invalid release tag.")
        release = self.json("/releases/tags/" + urllib.parse.quote(tag, safe=""))
        require(isinstance(release, dict) and release.get("tag_name") == tag
                and release.get("draft") is False and integer(release.get("id"), 1), "Release identity/status mismatch.")
        assets = {}
        for page in range(1, 102):
            batch = self.json(f"/releases/{release['id']}/assets?per_page=100&page={page}")
            require(isinstance(batch, list) and len(batch) <= 100, "Invalid release assets response.")
            for asset in batch:
                require(isinstance(asset, dict) and isinstance(asset.get("name"), str)
                        and asset["name"] not in assets and integer(asset.get("id"), 1)
                        and integer(asset.get("size")) and asset.get("state") == "uploaded", "Ambiguous release asset.")
                assets[asset["name"]] = asset
            if len(batch) < 100:
                return assets
        raise DownloadError("Too many release assets.")

    def asset(self, record):
        # Never consume a URL supplied in remote metadata.
        return self.open(f"/releases/assets/{record['id']}", binary=True)


def copy_verified(source, target, spec):
    digest, total = hashlib.sha256(), 0
    while True:
        block = source.read(CHUNK)
        if not block:
            break
        total += len(block)
        require(total <= spec["size"], "Stream exceeds pinned byte size.")
        digest.update(block)
        target.write(block)
    require(total == spec["size"] and digest.hexdigest() == spec["sha256"], "Downloaded/extracted SHA256 or size mismatch.")


def publish_no_replace(source, target):
    """Atomic no-replace publication on the same filesystem; stage is removed later."""
    target = plain_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    plain_path(target.parent)
    try:
        os.link(source, target)
    except FileExistsError:
        raise DownloadError("Destination appeared concurrently; refusing to replace it.") from None


def download_shard(client, asset, shard, cache):
    require(asset["size"] == shard["size"], "Release shard size differs from pinned index.")
    final = cache / shard["name"]
    if exact_file(final, shard):
        return final, True
    partial = plain_path(cache / (shard["name"] + ".partial"))
    if partial.exists():
        require(partial.is_file() and partial.stat().st_nlink == 1, "Unsafe partial download path.")
    with client.asset(asset) as source, partial.open("wb") as target:
        copy_verified(source, target, shard)
    publish_no_replace(partial, final)
    partial.unlink()
    return final, False


def install_shard(archive, shard, root, cache):
    require(exact_file(archive, shard), "Archive missing.")
    expected = {item["path"]: item for item in shard["files"]}
    existing = {name for name, item in expected.items() if exact_file(root / name, item)}
    if len(existing) == len(expected):
        return 0
    # Only our randomly named, exclusively created staging directory is removed.
    with tempfile.TemporaryDirectory(prefix="extract-", dir=cache) as temporary:
        staging, seen = Path(temporary), set()
        try:
            with tarfile.open(archive, mode="r|gz") as bundle:
                for member in bundle:
                    name = safe_member(member.name)
                    require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and not member.issparse(),
                            "Archive links, directories, devices and sparse files are forbidden.")
                    require(name in expected and name not in seen, "Unexpected or duplicate archive member.")
                    seen.add(name)
                    require(member.size == expected[name]["size"], "Archive member size mismatch.")
                    staged = staging / name
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.extractfile(member) as source, staged.open("xb") as target:
                        copy_verified(source, target, expected[name])
            require(seen == set(expected), "Archive is missing pinned members.")
        except (tarfile.TarError, EOFError):
            raise DownloadError("Invalid compressed archive.") from None
        # Validate ALL members before publishing ANY member of this shard.
        for name in expected:
            if name not in existing:
                publish_no_replace(staging / name, root / name)
    return len(expected) - len(existing)


def verify_install(index, destination):
    root = plain_path(destination)
    for shard in index["shards"]:
        for item in shard["files"]:
            require(exact_file(root / item["path"], item), "Installed data file missing.")
    return {"status": "verified", "files": index["total_files"], "bytes": index["total_bytes"]}


def download(index, pinned_bytes, client, tag, destination, progress=print):
    validate_index(index)
    require(parse_json(pinned_bytes) == index, "Pinned index bytes do not match parsed index.")
    assets = client.release_assets(tag)
    require(INDEX_ASSET in assets, "Release is missing its data index.")
    index_asset = assets[INDEX_ASSET]
    require(index_asset["size"] == len(pinned_bytes), "Release index differs from the cloned index.")
    with client.asset(index_asset) as stream:
        remote_index = stream.read(len(pinned_bytes) + 1)
    require(remote_index == pinned_bytes, "Release index differs from the cloned index; refuse unpinned data.")
    require(all(shard["name"] in assets for shard in index["shards"]), "Release is missing pinned shards.")
    root = plain_path(destination)
    root.mkdir(parents=True, exist_ok=True)
    cache = create_cache(root)
    # Conservative capacity estimate, excluding already verified destination files.
    needed, largest_stage = 0, 0
    for shard in index["shards"]:
        if not exact_file(cache / shard["name"], shard):
            needed += shard["size"]
        has_missing = False
        for item in shard["files"]:
            if not exact_file(root / item["path"], item):
                needed += item["size"]
                has_missing = True
        if has_missing:
            largest_stage = max(largest_stage, sum(item["size"] for item in shard["files"]))
    needed += largest_stage
    require(shutil.disk_usage(root).free >= needed, "Insufficient free disk for cache, installed files and one staging shard.")
    reused, installed = 0, 0
    for shard in index["shards"]:
        archive, skip = download_shard(client, assets[shard["name"]], shard, cache)
        installed += install_shard(archive, shard, root, cache)
        reused += int(skip)
        progress(f"Verified {shard['name']} ({'cached' if skip else 'downloaded'})")
    return {"status": "completed", "bundle_id": index["bundle_id"], "files": index["total_files"],
            "new_files": installed, "shards": len(index["shards"]), "reused_shards": reused}


def main(argv=None):
    home = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--index", type=Path, default=home / INDEX_ASSET)
    parser.add_argument("--destination", type=Path, default=home)
    parser.add_argument("--authenticated", action="store_true",
                        help="Explicitly use GH_TOKEN/GITHUB_TOKEN or gh auth token; default is anonymous.")
    parser.add_argument("--verify-only", action="store_true", help="Verify installed files locally; no token or network.")
    args = parser.parse_args(argv)
    try:
        index, raw = load_index(args.index)
        result = (verify_install(index, args.destination) if args.verify_only else
                  download(index, raw, GitHub(args.repo, get_token() if args.authenticated else None),
                           args.tag, args.destination))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except DownloadError as exc:
        print("Data error: " + str(exc), file=sys.stderr)
        return 1
    except (OSError, ValueError, tarfile.TarError):
        # Network exceptions can contain signed redirect URLs; do not print them.
        print("Data transfer or filesystem operation failed; completed shards are retained. No automatic retry.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
