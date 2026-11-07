"""Tiny ZIP fixtures; no actual corpus, frozen suite, model, or GPU is needed."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import hydrate_artifact as hydrate


ASSETS = {"data/train.jsonl": b'{"sample":1}\n', "frozen/example.py": b"VALUE = 1\n"}


def record(path, content):
    return {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}


def fixture(base: Path, *, asset_bytes=None, extras=()):
    base.mkdir(parents=True, exist_ok=True)
    root = base / "repo"
    root.mkdir()
    manifest = {"version": 1, "files": [record(path, content) for path, content in ASSETS.items()] +
                [record("README.md", b"checkout documentation\n"), record("run.py", b"checkout code\n")]}
    (root / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    archive = base / "artifact.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path, content in (asset_bytes if asset_bytes is not None else ASSETS).items():
            bundle.writestr(path, content)
        for path, content in extras:
            bundle.writestr(path, content)
    return root, archive


@unittest.skipUnless(os.name == "posix", "The documented runtime is Colab/Linux or Ubuntu CI")
class HydrationChecks(unittest.TestCase):
    def test_only_assets_are_restored_and_identical_files_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root, archive = fixture(Path(directory), extras=[
                ("README.md", b"archive documentation that must not replace checkout\n"),
                ("run.py", b"archive code that must never be installed\n"),
                ("PACKAGE_MANIFEST.json", b"archive manifest is not trusted"),
            ])
            (root / "README.md").write_bytes(b"checkout documentation\n")
            first = hydrate.hydrate_archive(archive, root)
            self.assertEqual(first["restored"], 2)
            self.assertEqual(first["already_matching"], 0)
            for path, expected in ASSETS.items():
                self.assertEqual((root / path).read_bytes(), expected)
            self.assertEqual((root / "README.md").read_bytes(), b"checkout documentation\n")
            self.assertFalse((root / "run.py").exists())
            before = {path: (root / path).stat().st_mtime_ns for path in ASSETS}
            second = hydrate.hydrate_archive(archive, root)
            self.assertEqual(second["restored"], 0)
            self.assertEqual(second["already_matching"], 2)
            self.assertEqual(before, {path: (root / path).stat().st_mtime_ns for path in ASSETS})
            self.assertFalse(list(root.rglob(".hydrate-*.tmp")))

    def test_check_only_creates_no_directories_or_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root, archive = fixture(Path(directory))
            before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
            result = hydrate.hydrate_archive(archive, root, check_only=True)
            self.assertEqual(result["state"], "verified_only")
            self.assertEqual(result["missing_before"], 2)
            self.assertEqual(result["restored"], 0)
            self.assertEqual(result["remaining_missing"], 2)
            self.assertEqual(before, sorted(path.relative_to(root).as_posix() for path in root.rglob("*")))

    def test_existing_mismatch_rejects_before_restoring_any_other_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root, archive = fixture(Path(directory))
            (root / "data").mkdir()
            original = b"user data must survive"
            (root / "data/train.jsonl").write_bytes(original)
            with self.assertRaisesRegex(hydrate.HydrationError, "refusing overwrite"):
                hydrate.hydrate_archive(archive, root)
            self.assertEqual((root / "data/train.jsonl").read_bytes(), original)
            self.assertFalse((root / "frozen").exists())

    def test_late_bad_hash_is_rejected_before_first_install(self):
        with tempfile.TemporaryDirectory() as directory:
            corrupt = dict(ASSETS)
            corrupt["frozen/example.py"] = b"VALUE = 2\n"  # same size, different SHA
            root, archive = fixture(Path(directory), asset_bytes=corrupt)
            with self.assertRaisesRegex(hydrate.HydrationError, "SHA-256 mismatch"):
                hydrate.hydrate_archive(archive, root)
            self.assertFalse((root / "data").exists())
            self.assertFalse((root / "frozen").exists())

    def test_unsafe_zip_paths_are_rejected_even_outside_asset_prefixes(self):
        bad_paths = ["../escape", "/absolute", "data/../../escape", "data\\escape", "data//file", "data/./file", "C:drivefile"]
        with tempfile.TemporaryDirectory() as directory:
            for index, bad in enumerate(bad_paths):
                with self.subTest(path=bad):
                    root, archive = fixture(Path(directory) / str(index), extras=[(bad, b"bad")])
                    with self.assertRaises(hydrate.HydrationError):
                        hydrate.hydrate_archive(archive, root)
                    self.assertFalse((root / "data").exists())
            self.assertFalse((Path(directory) / "escape").exists())

    def test_duplicate_members_and_zip_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                root, archive = fixture(Path(directory) / "duplicate", extras=[("data/train.jsonl", ASSETS["data/train.jsonl"])])
            with self.assertRaisesRegex(hydrate.HydrationError, "Duplicate"):
                hydrate.hydrate_archive(archive, root)
            self.assertFalse((root / "data").exists())
            root, archive = fixture(Path(directory) / "symlink")
            link = zipfile.ZipInfo("unrelated_link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "a") as bundle:
                bundle.writestr(link, "../../elsewhere")
            with self.assertRaisesRegex(hydrate.HydrationError, "ZIP symlink"):
                hydrate.hydrate_archive(archive, root)
            self.assertFalse((root / "data").exists())

    def test_existing_destination_symlinks_are_never_followed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, archive = fixture(base / "parent_link")
            outside = base / "outside"
            outside.mkdir()
            (root / "data").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(hydrate.HydrationError, "symlink"):
                hydrate.hydrate_archive(archive, root)
            self.assertEqual(list(outside.iterdir()), [])
            root, archive = fixture(base / "file_link")
            (root / "data").mkdir()
            outside_file = outside / "user.jsonl"
            outside_file.write_bytes(ASSETS["data/train.jsonl"])
            (root / "data/train.jsonl").symlink_to(outside_file)
            with self.assertRaisesRegex(hydrate.HydrationError, "symlink"):
                hydrate.hydrate_archive(archive, root)
            self.assertEqual(outside_file.read_bytes(), ASSETS["data/train.jsonl"])
            self.assertFalse((root / "frozen").exists())

    def test_manifest_self_reference_and_undeclared_assets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, archive = fixture(Path(directory) / "self")
            path = root / "PACKAGE_MANIFEST.json"
            content = json.loads(path.read_text())
            content["files"].append(record("PACKAGE_MANIFEST.json", b"self"))
            path.write_text(json.dumps(content))
            with self.assertRaisesRegex(hydrate.HydrationError, "exclude itself"):
                hydrate.hydrate_archive(archive, root)
            root, archive = fixture(Path(directory) / "extra", extras=[("data/undeclared.json", b"not in manifest")])
            with self.assertRaisesRegex(hydrate.HydrationError, "unlisted asset"):
                hydrate.hydrate_archive(archive, root)
            self.assertFalse((root / "data").exists())

    def test_enclosing_archive_folder_is_not_silently_stripped(self):
        with tempfile.TemporaryDirectory() as directory:
            root, archive = fixture(Path(directory), asset_bytes={"package/" + key: value for key, value in ASSETS.items()})
            with self.assertRaisesRegex(hydrate.HydrationError, "no enclosing folder"):
                hydrate.hydrate_archive(archive, root)
            self.assertFalse((root / "data").exists())

    def test_atomic_install_never_replaces_a_racing_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root, archive = fixture(Path(directory))
            (root / "data").mkdir()
            parent_fd = os.open(root / "data", os.O_RDONLY | os.O_DIRECTORY)
            competing = b"another process wrote this"

            def racing_link(source, destination, **kwargs):
                descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644, dir_fd=kwargs["dst_dir_fd"])
                try:
                    os.write(descriptor, competing)
                finally:
                    os.close(descriptor)
                raise FileExistsError(destination)

            try:
                with zipfile.ZipFile(archive) as bundle, patch.object(hydrate.os, "link", side_effect=racing_link):
                    expected = record("data/train.jsonl", ASSETS["data/train.jsonl"])
                    with self.assertRaisesRegex(hydrate.HydrationError, "refusing overwrite"):
                        hydrate._install_new(bundle, bundle.getinfo(expected["path"]), expected, parent_fd)
            finally:
                os.close(parent_fd)
            self.assertEqual((root / "data/train.jsonl").read_bytes(), competing)
            self.assertFalse(list((root / "data").glob(".hydrate-*.tmp")))


if __name__ == "__main__":
    unittest.main()
