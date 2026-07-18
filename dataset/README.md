# Dataset

This directory holds the fashion product images that the indexer embeds.

The original dataset for the Glance ML Internship Assignment is
[Fashionpedia](https://fashionpedia.github.io/home/Fashionpedia_download.html).
A reasonable subset of **≥ 1,000 images** across the three required axes
(environment, clothing type, color) is sufficient.

## Expected layout

```
dataset/
├── README.md        ← this file
└── images/
    ├── abc123.jpg   ← any filenames; the indexer sorts and processes all
    ├── def456.jpg
    └── ... (≥ 1,000 images)
```

## How to populate

1. Download a Fashionpedia subset (or any fashion image dataset).
2. Place images as `.jpg` / `.jpeg` / `.png` / `.webp` / `.bmp` directly in `images/`.
3. Run the indexer:

   ```powershell
   python indexer/build_index.py
   ```

## Notes

- The indexer discovers images by extension; filenames can be anything.
- ~3,200 images is what the development run used. The pipeline scales to ≥ 1 M images via `IndexIVFFlat`.
- This folder is **gitignored** because image corpora are large and binary; clone the repo, populate locally, index locally.