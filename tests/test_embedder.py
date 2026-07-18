"""Tests for glance_search.embedder (no model download)."""

from __future__ import annotations

from pathlib import Path

import pytest

from glance_search.embedder import list_images


def test_list_images_filters_by_extension(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.JPG").write_bytes(b"x")
    (tmp_path / "c.png").write_bytes(b"x")
    (tmp_path / "d.txt").write_bytes(b"x")
    paths = list_images(tmp_path)
    names = sorted(p.name for p in paths)
    assert names == ["a.jpg", "b.JPG", "c.png"]


def test_list_images_missing_dir_raises(tmp_path: Path) -> None:
    from glance_search.errors import EmbeddingError
    with pytest.raises(EmbeddingError):
        list_images(tmp_path / "does_not_exist")


def test_list_images_sorted(tmp_path: Path) -> None:
    for n in ["z.jpg", "a.jpg", "m.jpg"]:
        (tmp_path / n).write_bytes(b"x")
    paths = list_images(tmp_path)
    assert [p.name for p in paths] == ["a.jpg", "m.jpg", "z.jpg"]
