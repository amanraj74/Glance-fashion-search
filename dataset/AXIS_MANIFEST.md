# Dataset Axis Manifest

The five rubric queries are designed to probe different *axes* of compositional
fashion retrieval. This manifest documents which axis each query targets so that
the ablation results can be interpreted axis-by-axis.

## Axes

| Axis | What it tests | Example hit |
|---|---|---|
| **color** | Single-attribute colour matching | "yellow raincoat" |
| **garment** | Specific clothing item | "blue shirt" |
| **scene** | Location / setting | "park bench", "modern office" |
| **style** | Stylistic inference (no specific garment/colour) | "casual weekend outfit" |
| **material** | Fabric / texture | "silk", "denim" |
| **compositional** | Binding two attributes together | "red tie + white shirt" |
| **contextual** | Garment + location together | "blue shirt + park bench" |
| **single-attribute** | One strong visual concept | "yellow raincoat" |

## The 5 rubric queries

| Slug | Query | Axis tags | Difficulty |
|---|---|---|---|
| `01_yellow_raincoat` | A person in a bright yellow raincoat. | color, garment, single-attribute | Easy |
| `02_business_office` | Professional business attire inside a modern office. | garment, scene, contextual | Medium |
| `03_blue_shirt_park` | Someone wearing a blue shirt sitting on a park bench. | color, garment, scene, contextual | Medium |
| `04_casual_city` | Casual weekend outfit for a city walk. | style, scene | Hard (no specific colour/garment) |
| `05_red_tie_white_shirt` | A red tie and a white shirt in a formal setting. | color, garment, compositional | Hard (compositional) |

## Why this matters

A system that scores 0.5 mean on all five queries is doing well overall but might
actually be failing the *hard* axes (compositional, style). This manifest lets
the eval report break out per-axis performance so that wins and regressions can
be attributed to the right component (encoder, captions, reranker, etc.).

## Coverage of the catalogue

The catalogue (`dataset/images/`, 3,200 product images) is fashion-finetuned
data; it skews toward clothing on plain backgrounds. By BLIP-base caption
analysis:

- 40.8 % of captions contain "runway" (runway shows dominate the catalogue)
- 51.1 % contain at least one colour word
- 54.0 % contain at least one garment word
- 100 % have *some* text description

This explains why Q4 ("casual city") — which has no specific colour or garment
in the query — is the hardest: the catalogue lacks "casual" data and the
captions skew toward runway vocabulary.

## How to regenerate

The axis tags are computed dynamically by `glance_search.attributes.parse_query`
and embedded in `eval/results/metrics.csv`. The table above is a manual
documentation layer that adds the difficulty and rationale.
