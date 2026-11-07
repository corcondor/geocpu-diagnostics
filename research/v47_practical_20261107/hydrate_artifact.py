#!/usr/bin/env python3
"""Restore missing data/ and frozen/ assets from the separately supplied ZIP.

The checkout's PACKAGE_MANIFEST.json is authoritative. Archive code, docs,
tests, and its own manifest are never installed. The destination must be a
local POSIX checkout supporting no-follow directory opens and hard links;
this includes the intended Colab/Linux and Ubuntu CI environments.

Validation of all target bytes and existing files precedes every write.
Installation uses a completed temporary file plus an atomic, exclusive hard
link, so an existing file is never replaced, including a concurrent writer.
No Google Drive study path, checkpoint path, or training API is consulted.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import uuid
import zipfile


MANIFEST_NAME = "PACKAGE_MANIFEST.json"
ASSET_PREFIXES = frozenset({"data", "frozen"})
CHUNK_BYTES = 1024 * 1024


class HydrationError(RuntimeError):
    """An unsafe or inconsistent archive/checkout was rejected."""


def _safe_relative(value: object, *, directory: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise HydrationError("Manifest and ZIP paths must be nonempty strings")
    if "\\" in value or ":" in value or any(ord(char) < 32 for char in value):
        raise HydrationError(f"Unsafe path: {value!r}")
    text = value[:-1] if directory and value.endswith("/") else value
    if not text or text.startswith("/") or text.endswith("/"):
        raise HydrationError(f"Expected a canonical relative path: {value!r}")
    parts = text.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise HydrationError(f"Unsafe path component: {value!r}")
    if str(PurePosixPath(text)) != text:
        raise HydrationError(f"Noncanonical relative path: {value!r}")
    return text


def _asset_path(path: str) -> bool:
    parts = path.split("/")
    return len(parts) > 1 and parts[0] in ASSET_PREFIXES


def _platform_check() -> None:
    needed = (os.open, os.mkdir, os.unlink, os.link)
    if (os.name != "posix" or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY")
            or any(function not in os.supports_dir_fd for function in needed)):
        raise HydrationError("This hydrator requires POSIX no-follow directory operations; run it in Colab/Linux or Ubuntu CI")


def _read_manifest(root_fd: int) -> tuple[dict[str, dict], str]:
    try:
        descriptor = os.open(MANIFEST_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
    except OSError as error:
        raise HydrationError("The checkout must contain a regular, non-symlink PACKAGE_MANIFEST.json") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise HydrationError("PACKAGE_MANIFEST.json must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            blob = stream.read()
    finally:
        os.close(descriptor)
    try:
        document = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HydrationError("PACKAGE_MANIFEST.json is not valid JSON") from error
    if (not isinstance(document, dict) or type(document.get("version")) is not int or document["version"] != 1
            or not isinstance(document.get("files"), list)):
        raise HydrationError("Expected manifest schema version 1 with a files array")
    records, casefold_paths = {}, set()
    for record in document["files"]:
        if not isinstance(record, dict):
            raise HydrationError("Each manifest file must be an object")
        path = _safe_relative(record.get("path"))
        if path == MANIFEST_NAME:
            raise HydrationError("PACKAGE_MANIFEST.json must exclude itself")
        folded = path.casefold()
        if path in records or folded in casefold_paths:
            raise HydrationError(f"Duplicate or case-colliding manifest path: {path}")
        digest, size = record.get("sha256"), record.get("size_bytes")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise HydrationError(f"Invalid SHA-256 for {path}")
        if type(size) is not int or size < 0:
            raise HydrationError(f"Invalid size_bytes for {path}")
        records[path] = {"path": path, "sha256": digest.lower(), "size_bytes": size}
        casefold_paths.add(folded)
    # A file cannot also be an ancestor directory of another declared file.
    folded_set = {path.casefold() for path in records}
    for path in records:
        parts = path.split("/")
        if any("/".join(parts[:index]).casefold() in folded_set for index in range(1, len(parts))):
            raise HydrationError(f"Manifest file/directory collision: {path}")
    return records, hashlib.sha256(blob).hexdigest()


def _archive_index(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    files, all_entries = {}, {}
    for info in archive.infolist():
        # ZipInfo can truncate names at NUL. Do not accept a silently changed name.
        if info.orig_filename != info.filename:
            raise HydrationError("ZIP contains a truncated or changed filename")
        directory = info.is_dir()
        path = _safe_relative(info.filename, directory=directory)
        key = path.casefold()
        if key in all_entries:
            raise HydrationError(f"Duplicate or case-colliding ZIP path: {path}")
        mode = info.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if stat.S_ISLNK(mode):
            raise HydrationError(f"ZIP symlink rejected: {path}")
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise HydrationError(f"ZIP special file rejected: {path}")
        if (kind == stat.S_IFDIR and not directory) or (kind == stat.S_IFREG and directory):
            raise HydrationError(f"ZIP file type disagrees with path: {path}")
        if info.flag_bits & 1:
            raise HydrationError(f"Encrypted ZIP member is unsupported: {path}")
        all_entries[key] = (path, directory)
        if not directory:
            files[path] = info
    for path, _ in all_entries.values():
        parts = path.split("/")
        for index in range(1, len(parts)):
            parent = all_entries.get("/".join(parts[:index]).casefold())
            if parent is not None and not parent[1]:
                raise HydrationError(f"ZIP file/directory collision: {path}")
    return files


def _stream_digest(stream, expected: dict, sink=None) -> None:
    digest, count = hashlib.sha256(), 0
    while True:
        chunk = stream.read(CHUNK_BYTES)
        if not chunk:
            break
        count += len(chunk)
        if count > expected["size_bytes"]:
            raise HydrationError(f"Size exceeds manifest: {expected['path']}")
        digest.update(chunk)
        if sink is not None:
            sink.write(chunk)
    if count != expected["size_bytes"] or digest.hexdigest() != expected["sha256"]:
        raise HydrationError(f"Size or SHA-256 mismatch: {expected['path']}")


@contextmanager
def _parent_directory(root_fd: int, path: str, *, create: bool):
    descriptor = os.dup(root_fd)
    try:
        for component in path.split("/")[:-1]:
            try:
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    yield None
                    return
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass  # Another writer may have created it; the no-follow open checks its type.
                try:
                    child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                except OSError as error:
                    raise HydrationError(f"Destination parent is not a regular directory: {path}") from error
            except OSError as error:
                raise HydrationError(f"Destination symlink or non-directory rejected: {path}") from error
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _existing_matches(parent_fd: int | None, expected: dict) -> bool:
    if parent_fd is None:
        return False
    basename = expected["path"].rsplit("/", 1)[-1]
    try:
        descriptor = os.open(basename, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0), dir_fd=parent_fd)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise HydrationError(f"Destination symlink or inaccessible file rejected: {expected['path']}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise HydrationError(f"Destination must be a regular file: {expected['path']}")
        if metadata.st_size != expected["size_bytes"]:
            raise HydrationError(f"Existing destination differs; refusing overwrite: {expected['path']}")
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                _stream_digest(stream, expected)
        except HydrationError as error:
            raise HydrationError(f"Existing destination differs; refusing overwrite: {expected['path']}") from error
    finally:
        os.close(descriptor)
    return True


def _install_new(archive: zipfile.ZipFile, info: zipfile.ZipInfo, expected: dict, parent_fd: int) -> bool:
    """Return True if installed, False if an identical concurrent file won."""
    if _existing_matches(parent_fd, expected):
        return False
    basename = expected["path"].rsplit("/", 1)[-1]
    temporary = ".hydrate-" + uuid.uuid4().hex + ".tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as sink:
            with archive.open(info, "r") as source:
                _stream_digest(source, expected, sink)
            sink.flush()
            os.fchmod(descriptor, 0o644)
            os.fsync(descriptor)
        try:
            # link() atomically creates a new name and never replaces one.
            os.link(temporary, basename, src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
        except FileExistsError:
            if _existing_matches(parent_fd, expected):
                return False
            raise HydrationError(f"Destination appeared during installation: {expected['path']}")
        except OSError as error:
            raise HydrationError("Atomic exclusive installation requires a local filesystem with hard-link support") from error
        os.fsync(parent_fd)
        return True
    finally:
        os.close(descriptor)
        os.unlink(temporary, dir_fd=parent_fd)


def hydrate_archive(archive_path: Path | str, root: Path | str, *, check_only: bool = False) -> dict:
    """Validate an archive, then restore only missing manifest-listed assets.

    check_only validates archive bytes and existing targets and returns the
    number that would be restored. It does not create files or directories.
    Manifest-listed code/docs/tests are outside this function's write scope.
    """
    _platform_check()
    root = Path(root).absolute()
    archive_path = Path(archive_path).absolute()
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise HydrationError("The root must be an existing, non-symlink checkout directory") from error
    try:
        records, manifest_sha = _read_manifest(root_fd)
        assets = {path: metadata for path, metadata in records.items() if _asset_path(path)}
        if not assets:
            raise HydrationError("Manifest contains no data/ or frozen/ file assets")
        try:
            archive_fd = os.open(archive_path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
        except OSError as error:
            raise HydrationError("Archive must be an accessible regular, non-symlink file") from error
        try:
            if not stat.S_ISREG(os.fstat(archive_fd).st_mode):
                raise HydrationError("Archive must be a regular file")
            with os.fdopen(archive_fd, "rb", closefd=False) as archive_stream, zipfile.ZipFile(archive_stream) as archive:
                index = _archive_index(archive)
                extras = [path for path in index if _asset_path(path) and path not in assets]
                if extras:
                    raise HydrationError(f"Archive has unlisted asset files: {extras[:3]}")
                missing_members = sorted(set(assets) - set(index))
                if missing_members:
                    raise HydrationError("Archive lacks manifest assets at its root (no enclosing folder): " + str(missing_members[:3]))
                # A complete preflight prevents a late corrupt member from
                # causing a partially hydrated checkout.
                for path, expected in assets.items():
                    if index[path].file_size != expected["size_bytes"]:
                        raise HydrationError(f"Archive size differs from manifest: {path}")
                    with archive.open(index[path], "r") as source:
                        _stream_digest(source, expected)
                missing, matching = [], []
                for path, expected in assets.items():
                    with _parent_directory(root_fd, path, create=False) as parent_fd:
                        (matching if _existing_matches(parent_fd, expected) else missing).append(path)
                restored = []
                if not check_only:
                    for path in missing:
                        with _parent_directory(root_fd, path, create=True) as parent_fd:
                            if _install_new(archive, index[path], assets[path], parent_fd):
                                restored.append(path)
                            else:
                                matching.append(path)
                return {"state": "verified_only" if check_only else "complete", "manifest_sha256": manifest_sha,
                        "asset_files": len(assets), "archive_files_checked_for_safe_paths": len(index),
                        "ignored_non_asset_files": sum(not _asset_path(path) for path in index),
                        "missing_before": len(missing), "restored": len(restored), "already_matching": len(matching),
                        "remaining_missing": len(missing) if check_only else 0,
                        "restored_paths": restored, "check_only": check_only}
        except (zipfile.BadZipFile, NotImplementedError, RuntimeError, EOFError) as error:
            if isinstance(error, HydrationError):
                raise
            raise HydrationError("Archive validation or restoration failed: " + str(error)) from error
        finally:
            os.close(archive_fd)
    finally:
        os.close(root_fd)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Artifact ZIP with data/ and frozen/ directly at archive root")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Checkout containing its authoritative PACKAGE_MANIFEST.json")
    parser.add_argument("--check-only", action="store_true", help="Validate and report planned missing assets without any writes")
    args = parser.parse_args(argv)
    try:
        result = hydrate_archive(args.archive, args.root, check_only=args.check_only)
    except (HydrationError, OSError) as error:
        print("HYDRATION_REJECTED: " + str(error), file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
