"""Metadata-only relocation of immutable source bytes; no trust bypass."""
from __future__ import annotations

import hashlib
import json
import ntpath
import posixpath
from pathlib import Path, PureWindowsPath, PurePosixPath


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def contained_path(root, relative):
    """Resolve portable relative paths, rejecting foreign absolute paths too."""
    root = Path(root).resolve()
    if (not isinstance(relative, str) or not relative or '\\' in relative
            or PureWindowsPath(relative).drive or Path(relative).is_absolute()):
        raise ValueError('Bundle paths must be portable relative paths')
    candidate = (root / relative).resolve()
    if root not in candidate.parents:
        raise ValueError('Bundle path escapes its root')
    return candidate


def bound_file(root, reference):
    path = contained_path(root, reference['path'])
    if file_sha256(path) != reference['sha256']:
        raise ValueError('Bundle file SHA256 mismatch: ' + reference['path'])
    return path


def _logical(path):
    value = str(path)
    windows = PureWindowsPath(value)
    if windows.drive:
        if not windows.is_absolute():
            raise ValueError('Drive-relative source paths are forbidden')
        return PureWindowsPath(ntpath.normpath(value))
    posix = PurePosixPath(value)
    if not posix.is_absolute():
        raise ValueError('Original source identity must be absolute')
    return PurePosixPath(posixpath.normpath(value))


def _key(path):
    value = _logical(path)
    return str(value).casefold() if isinstance(value, PureWindowsPath) else str(value)


class PortableSources:
    """Map original path identities to hash-verified local immutable files.

    Original paths are lookup keys only. Missing keys never fall back to disk.
    Original JSON and scientific provenance comparisons remain unchanged.
    """
    def __init__(self, wrapper_path, document=None):
        self.root = Path(wrapper_path).resolve().parent
        document = read_json(wrapper_path) if document is None else document
        if document.get('schema_version') != 'fwt_power_portable_manifest_v1':
            raise ValueError('Expected portable POWER source map')
        original = document['original_manifest']
        self.default_base = _logical(original['original_path']).parent
        self.mapping = {}
        files = document.get('files')
        if not isinstance(files, list) or not files:
            raise ValueError('Portable source map is empty')
        for reference in files:
            key = _key(reference['original_path'])
            if key in self.mapping:
                raise ValueError('Duplicate original source identity')
            path = bound_file(self.root, reference)
            if path.suffix.lower() not in ('.json', '.py', '.txt'):
                raise ValueError('Source evidence must be JSON, script or HTTP headers')
            self.mapping[key] = (path, reference['sha256'])
        if self.mapping.get(_key(original['original_path'])) != (
                bound_file(self.root, original), original['sha256']):
            raise ValueError('Original manifest must be included in the source map')

    def logical_path(self, source, base):
        source = str(source)
        if PureWindowsPath(source).drive or PurePosixPath(source).is_absolute():
            return _logical(source)
        # Content addressing changes physical locations, not original bases.
        if isinstance(base, PureWindowsPath) and not isinstance(base, Path):
            logical_base = base
        elif isinstance(base, PurePosixPath) and not isinstance(base, Path):
            logical_base = base
        else:
            logical_base = self.default_base
        return _logical(logical_base / source)

    def resolve(self, source, base):
        key = _key(self.logical_path(source, base))
        if key not in self.mapping:
            raise ValueError('Unmapped immutable source path: ' + str(source))
        return self.mapping[key][0]

    def logical_parent(self, source, base):
        return self.logical_path(source, base).parent
