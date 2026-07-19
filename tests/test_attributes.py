"""Tests for glance_search.attributes — no model download required."""

from __future__ import annotations

from glance_search.attributes import (
    attribute_overlap_score,
    missing_attributes,
    parse_query,
    query_axis_tags,
)


def test_parse_query_extracts_color() -> None:
    a = parse_query("a bright yellow raincoat")
    assert "yellow" in a.colors


def test_parse_query_extracts_garment() -> None:
    a = parse_query("red tie and white shirt")
    assert "tie" in a.garments
    assert "shirt" in a.garments


def test_parse_query_extracts_scene() -> None:
    a = parse_query("casual weekend outfit for a city walk")
    assert "city" in a.scenes


def test_parse_query_extracts_style() -> None:
    a = parse_query("elegant vintage wedding dress")
    assert "vintage" in a.styles


def test_parse_query_total_hits() -> None:
    a = parse_query("yellow raincoat in a park")
    assert a.total_hits >= 2


def test_parse_query_free_form_returns_generic_tag() -> None:
    a = parse_query("lorem ipsum dolor sit amet")
    assert query_axis_tags(a) == ["free-form"]


def test_attribute_overlap_score_zero_when_empty() -> None:
    assert attribute_overlap_score(parse_query("yellow raincoat"), "") == 0.0
    assert attribute_overlap_score(parse_query("yellow raincoat"), None) == 0.0


def test_attribute_overlap_score_positive_when_match() -> None:
    a = parse_query("yellow raincoat in a park")
    cap = "a woman wearing a yellow raincoat sitting on a park bench"
    score = attribute_overlap_score(a, cap)
    assert score > 0.5


def test_attribute_overlap_score_partial() -> None:
    a = parse_query("yellow raincoat in a park")
    cap = "a woman in a yellow coat"  # missing park
    score = attribute_overlap_score(a, cap)
    assert 0.0 < score < 1.0


def test_missing_attributes_lists_uncovered() -> None:
    a = parse_query("red tie and white shirt in a park")
    cap = "a man in a suit and red tie"
    missing = missing_attributes(a, cap)
    assert "red" not in missing  # matched
    assert "tie" not in missing  # matched
    assert "white" in missing
    assert "shirt" in missing


def test_query_axis_tags_returns_list() -> None:
    a = parse_query("yellow shirt in a park")
    tags = query_axis_tags(a)
    assert "color" in tags
    assert "garment" in tags
    assert "scene" in tags