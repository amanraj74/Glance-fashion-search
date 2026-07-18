# How to Run This Project — Step by Step

A plain-English walkthrough that takes you from a fresh clone to a working search engine. Works on Windows, macOS, and Linux.

**Estimated total time:** ~90 minutes on first run (most of it is model downloads and CPU indexing). Subsequent runs take seconds.

---

## Before you start

Make sure you have these:

| Requirement | How to check | What you need |
|---|---|---|
| Python | `python --version` | **3.10 or newer** |
| pip | `pip --version` | bundled with Python |
| Git | `git --version` | any recent version |
| Internet | — | to download model weights (~1.5 GB total on first run) |
| Disk space | — | ~3 GB (models + index + catalogue) |
| RAM | — | 4 GB minimum, 8 GB recommended |

If you're on Windows, you'll be using PowerShell. On macOS or Linux, use the equivalent bash commands (they're shown alongside).

---

## Step 1 — Clone the repository

```powershell
git clone https://github.com/amanraj74/Glance-fashion-search.git
cd Glance-fashion-search
```

macOS / Linux:

```bash
git clone https://github.com/amanraj74/Glance-fashion-search.git
cd Glance-fashion-search
```

You should see these top-level files and folders:

```
Glance-fashion-search/
├── README.md
├── LICENSE
├── config.yaml
├── requirements.txt
├── pyproject.toml
├── src/
├── indexer/
├── retriever/
├── scripts/
├── app/
├── tests/
├── report/
├── dataset/        ← you should see images/ inside; if not, see Step 1b
└── output/         ← will be created in Step 4
```

### Step 1b — If the dataset is missing

The 3,200 product images are not committed to the repo (they're too big). If `dataset/images/` is empty, you need to add them. Easiest options:

1. **If you cloned from the original repo on GitHub**, the maintainer usually includes a `dataset/images.zip` link or instructions in the repo's README.
2. **Or generate a synthetic test set** by putting 10+ `.jpg` files into `dataset/images/` — the system will index and search them just the same.

---

## Step 2 — Create a virtual environment

This keeps the project's dependencies isolated from anything else on your machine.

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux (bash):**

```bash
python3 -m venv venv
source venv/bin/activate
```

You should now see `(venv)` at the start of your terminal prompt.

If PowerShell complains about execution policy on Windows, run this once and try again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Step 3 — Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest streamlit
```

Expected: a few minutes of downloading and compiling (PyTorch is the biggest). You should end with `Successfully installed ...` and no errors.

**Sanity check** — these should all import without errors:

```powershell
python -c "import torch, open_clip, faiss, sentence_transformers, transformers, streamlit; print('ok')"
```

Expected output: `ok`.

---

## Step 4 — Build the image index

This embeds every image in `dataset/images/` using a fashion-aware CLIP model and saves a FAISS index to `output/`.

```powershell
python indexer/build_index.py
```

**What you'll see:**

```
INFO loading model name=hf-hub:Marqo/marqo-fashionCLIP ...
INFO discovered 3200 images under dataset/images
embed: 100%|████████████████████| 200/200 [02:15<00:00,  1.48it/s]
INFO building IndexFlatIP n=3200 d=768
INFO wrote index=output\faiss.index ntotal=3200 metadata=output\metadata.json rows=3200
INFO done. index ntotal=3200 kept=3200
```

**Time:** ~20 minutes on CPU for 3,200 images. **Disk:** ~10 MB.

**If you only have a few test images:** the script handles small corpora fine — it just runs faster.

**If you have a GPU:** it will auto-detect CUDA and run much faster. Nothing to configure.

---

## Step 5 — Generate captions and build the caption index

This is the load-bearing step that makes compositional queries work. It runs BLIP over every image, writes a one-sentence caption to `output/captions.json`, and then embeds those captions with the same CLIP text encoder.

```powershell
python scripts/build_caption_index.py
```

**What you'll see:**

```
INFO loading caption model=Salesforce/blip-image-captioning-base ...
caption: 100%|████████████████████| 200/200 [08:30<00:00,  2.55s/it]
INFO wrote 3200 captions to output\captions.json
INFO building IndexFlatIP n=3200 d=768
INFO wrote index=output\captions.index ntotal=3200 metadata=output\caption_meta.json rows=3200
```

**Time:** ~45–75 minutes on CPU for 3,200 images. **Disk:** ~25 MB for captions JSON + ~10 MB for the index.

**Heads up:** this step is resumable. If you Ctrl-C in the middle, just run the command again — it'll pick up where it left off.

---

## Step 6 — Try a search

You're ready. Run your first query:

```powershell
python retriever/search.py --query "a person in a bright yellow raincoat" --top-k 5
```

Expected output:

```
Top results:

1. dataset\images\a4f9...c8.jpg | score=0.7455
2. dataset\images\8e5a...2b.jpg | score=0.6820
3. dataset\images\d20b...77.jpg | score=0.5884
4. dataset\images\1ae9...68.jpg | score=0.5712
5. dataset\images\7293...eb.jpg | score=0.5639
```

The score is in `[0, 1]` — higher means a closer match. Open any of the returned image paths to see the result.

**More queries to try** (all five are from the assignment brief):

```powershell
python retriever/search.py --query "Professional business attire inside a modern office"
python retriever/search.py --query "Someone wearing a blue shirt sitting on a park bench"
python retriever/search.py --query "Casual weekend outfit for a city walk"
python retriever/search.py --query "A red tie and a white shirt in a formal setting"
```

Or omit `--query` to be prompted interactively:

```powershell
python retriever/search.py
Enter your query: a black leather jacket
```

---

## Step 7 — Run the full evaluation

The assignment is judged on five specific queries. Run the harness:

```powershell
python scripts/run_eval.py
```

**What it does:** runs each of the 5 rubric queries, saves:

- `eval/results/01_yellow_raincoat.json` and `.png` — top-5 results as a grid image
- `eval/results/02_business_office.json` and `.png`
- `eval/results/03_blue_shirt_park.json` and `.png`
- `eval/results/04_casual_city.json` and `.png`
- `eval/results/05_red_tie_white_shirt.json` and `.png`
- `eval/results/summary.json` — per-query top scores

**Time:** under a minute.

Open the PNGs to eyeball the results. Open `summary.json` for the numbers:

```json
{
  "01_yellow_raincoat": { "query": "A person in a bright yellow raincoat.", "top_score": 0.7455 },
  "02_business_office": { "query": "Professional business attire inside a modern office.", "top_score": 0.3227 },
  ...
}
```

---

## Step 8 — Run the A/B ablation comparison

This compares three configurations of the pipeline on the five queries:

```powershell
python scripts/run_ablation.py
```

Output: `eval/results/ablation.csv` with rows like:

| config | query_slug | top1_score | top5_mean_score |
|---|---|---:|---:|
| image_only | 01_yellow_raincoat | 0.1029 | 0.0838 |
| image_captions | 01_yellow_raincoat | 0.3359 | 0.3336 |
| image_captions_rerank | 01_yellow_raincoat | 0.6596 | 0.3769 |
| ... | ... | ... | ... |

The three configurations are:

1. **`image_only`** — plain image-only retrieval (the vanilla CLIP-style baseline).
2. **`image_captions`** — adds caption similarity to the hybrid score.
3. **`image_captions_rerank`** — also re-ranks with a cross-encoder on top.

**Time:** under a minute.

---

## Step 9 — Build the writeup

The assignment deliverable is a PDF. Regenerate it from the markdown source:

```powershell
python scripts/build_report.py
```

Output:

```
wrote report\Glance_Internship_Report.html
wrote report\Glance_Internship_Report.pdf  (via fpdf2 - no GTK / system deps)
```

Two outputs:

- `report/Glance_Internship_Report.html` — open in any browser, looks the same as the PDF
- `report/Glance_Internship_Report.pdf` — the submission deliverable

**Heads up:** `weasyprint` would normally produce a higher-fidelity PDF, but it requires GTK system libraries that aren't on Windows. The script automatically falls back to `fpdf2`, which gives a clean text-and-tables output. If you want the weasyprint version, run this on a Linux machine or install GTK on Windows (advanced — not required).

**Time:** under 30 seconds.

---

## Step 10 — (Optional) Run the interactive web demo

If you want a browser-based UI:

```powershell
streamlit run app/streamlit_app.py
```

A browser opens at `http://localhost:8501` with:

- A text box for your query
- Sliders for **top-K**, **image weight**, **caption weight**
- A toggle for the cross-encoder re-ranker
- Per-result score breakdown (`image`, `caption`, `rerank`)
- The top-5 results as a grid

To stop the demo: Ctrl-C in the terminal.

---

## Step 11 — Run the tests

```powershell
pytest tests/ -v
```

Expected output: **21 passed in ~10 seconds**.

These tests are deliberately offline — they use synthetic embeddings and stub models, so they don't download anything. They're a good first check after cloning to make sure the code is wired up correctly.

---

## Quick command reference

```powershell
# Setup (once)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pytest streamlit

# Build indexes (once; ~1 hour total on CPU)
python indexer/build_index.py
python scripts/build_caption_index.py

# Daily use
python retriever/search.py --query "your fashion query here"
python scripts/run_eval.py              # 5-rubric evaluation
python scripts/run_ablation.py          # A/B comparison
streamlit run app/streamlit_app.py      # web demo

# Maintenance
pytest tests/ -v                        # test suite
python scripts/build_report.py          # rebuild report HTML/PDF
```

---

## Troubleshooting

**"No module named glance_search"**
You're not running from the repo root, or the venv isn't activated. `cd` into the cloned folder and run `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (macOS/Linux).

**"IndexNotFoundError: missing index artifacts"**
You skipped Step 4 or Step 5. Run `python indexer/build_index.py` and `python scripts/build_caption_index.py`.

**Search returns empty results**
The caption index is missing or empty. Check that `output/captions.json` exists and has entries. If empty, re-run Step 5.

**WeasyPrint error on Windows**
Expected. The script falls back to fpdf2 automatically and the output is fine.

**Model download is very slow**
Hugging Face is rate-limited or you're behind a slow network. Wait it out, or pre-download with `huggingface-cli download Marqo/marqo-fashionCLIP`.

**Out of memory during caption generation**
Edit `config.yaml` and lower `captions.batch_size` from 16 to 4 or 8. Re-run Step 5.

**Tests fail with `ModuleNotFoundError`**
Run `pip install pytest` to make sure pytest is in your venv.

---

## Where things live

```
output/                            ← generated by build scripts (gitignored)
├── faiss.index                    ← image index
├── metadata.json                  ← image paths, row-aligned with index
├── captions.json                  ← BLIP captions, one per image
├── captions.index                 ← caption embeddings index
└── caption_meta.json              ← caption index metadata

eval/results/                      ← generated by run_eval / run_ablation (gitignored)
├── 0N_*.png / .json               ← top-5 grids and raw results per query
├── summary.json                   ← per-query top scores
└── ablation.csv                   ← A/B comparison rows

report/                            ← generated by build_report
├── Glance_Internship_Report.md    ← markdown source (committed)
├── Glance_Internship_Report.html  ← rendered HTML (generated)
└── Glance_Internship_Report.pdf   ← rendered PDF (generated)
```

---

## Next steps

Once it's running, try:

1. **Tweak weights in `config.yaml`** — set `retrieval.image_weight: 0.7` and `retrieval.caption_weight: 0.3` to lean more on visual similarity, then re-run a query and see how the ranking changes.
2. **Disable the re-ranker** — set `retrieval.use_reranker: false` to compare the impact on latency vs. quality.
3. **Scale up the index** — switch to the approximate backend with `python indexer/build_index.py --backend ivfflat --ivf-nlist 4096`. Search stays fast on millions of vectors.
4. **Read the report** — `report/Glance_Internship_Report.md` has the full design rationale, evaluation results, and future-work roadmap.
