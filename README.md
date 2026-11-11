# LEGO Brick Recognition — Gruppe 2

**Photo in, brick list out, then: "What can I build?"**
Togzan, Ainur, Sami, David · Wintersemester 2026

No ready-made LEGO recogniser is used: segmentation, shape model and colour
classification are our own work.

```
 1 Capture ──► 2 Find bricks ──► 3a Shape (CNN)  ──► 4 Inventory ──► 5 What can I build?
 grey card     OpenCV            3b Colour (ΔE)      part·colour·#     Rebrickable part lists
 white bal.    (YOLO = stretch)  no neural net       + confidence      coverage + missing parts
```

Shape and colour are separate problems: **50 + 15 classes instead of 50 × 15 = 750**.

## Project layout

| Step | Code | What it does |
|---|---|---|
| 1 Capture | `lego/capture.py` | load photo, white balance with the grey card (in linear light) |
| 2 Find bricks | `lego/segment.py` | model the plain background as a smooth surface, threshold the Lab difference, clean up, split lightly touching bricks (watershed), cut square crops |
| 3a Shape | `lego/shape_cnn.py`, `lego/shape_baseline.py` | greyscale CNN (timm) with 4-rotation TTA · HOG + silhouette + SVM baseline |
| 3b Colour | `lego/colour.py` | robust median Lab of the brick pixels → nearest reference by CIEDE2000, "unsure" flag |
| 4 Inventory | `lego/inventory.py` | count per (part, colour); `expected` = confidence-weighted count |
| 5 Sets | `lego/sets.py` | coverage per set, missing parts, from Rebrickable dumps |
| render → real | `lego/augment.py` | `basic` vs `real` augmentation (blur, noise, JPEG, light, shadow, perspective, bad masks) |
| glue | `lego/pipeline.py`, `app.py` | end-to-end + Streamlit app with manual correction |

Scripts (`scripts/`): `train_cnn.py`, `train_baseline.py`, `evaluate.py`, `extract_crops.py`,
`split_real.py`, `calibrate_colours.py`, `prepare_rebrickable.py`, `run_pipeline.py`, `gradcam.py`.
Kaggle notebook: `notebooks/kaggle_train.ipynb`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_pipeline.py          # synthetic end-to-end test, no data needed
```

### 1. Data
* **Renders**: Kaggle dataset "Images of LEGO Bricks" (J. Hazelzet), 40,000 images, 50 classes.
  Easiest: train directly in a Kaggle notebook with the dataset attached.
* **Rebrickable**: download `sets`, `inventories`, `inventory_parts`, `colors` (`.csv.gz`) from
  rebrickable.com/downloads into `data/rebrickable/`, then
  `python scripts/prepare_rebrickable.py data/rebrickable`.
* **Our photos**: see the photo protocol in `docs/PLAN.md`.

### 2. Our photos → labelled crops
```bash
python scripts/extract_crops.py data/photos data/real_all     # one part type per photo = free labels
python scripts/split_real.py data/real_all data/real --test 0.3   # split BY PHOTO
```
Look through `data/real_all` once and delete bad crops (shadow only, two bricks).

### 3. Train
```bash
# baseline (CPU, minutes)
python scripts/train_baseline.py --renders data/renders --real-train data/real/train --real-test data/real/test

# CNN (Kaggle GPU) - the core experiment, see docs/PLAN.md
python scripts/train_cnn.py --renders /kaggle/input/images-of-lego-bricks \
    --real-test data/real/test --aug basic --tag A_renders_basic
python scripts/train_cnn.py --renders /kaggle/input/images-of-lego-bricks \
    --real-test data/real/test --aug real --tag B_renders_real
python scripts/train_cnn.py --renders /kaggle/input/images-of-lego-bricks \
    --real-train data/real/train --real-test data/real/test --aug real --tag C_finetuned
```
Every run appends a line to `results/experiments.jsonl`; `--wandb` logs to Weights & Biases.
Put the final model at `models/shape_cnn.pt`.

### 4. Evaluate (always on real photos)
```bash
python scripts/evaluate.py models/shape_cnn.pt data/real/test --tag C_finetuned
python scripts/gradcam.py models/shape_cnn.pt data/real/test
```
→ accuracy, top-3, macro-F1, confusion matrix, most-confused pairs, error gallery.

### 5. Colour calibration
```bash
python scripts/calibrate_colours.py data/colour_photos configs/colors_measured.csv
```
then set `colour: palette: configs/colors_measured.csv` in `configs/default.yaml`.

### 6. Use it
```bash
python scripts/run_pipeline.py photo.jpg --sets data/rebrickable/prepared.pkl
streamlit run app.py
```

## Configuration
`configs/default.yaml` — grey card position, segmentation thresholds, colour
thresholds, which shape model. If no CNN checkpoint / PyTorch is found the
pipeline falls back to the baseline, and without any model it still counts
bricks and colours.

## Open questions for the supervisors (slide 9)
The code supports either answer:
1. ImageNet-pretrained networks allowed? → `--pretrained` flag (default: off, trained from scratch).
2. YOLO with COCO weights for finding bricks? → only needed for piles (stretch goal); OpenCV path needs no weights.
3. Kaggle renders as training data? → if not, train only on `data/real/train` (same scripts).

## Honest limits
* Separated bricks on a plain background only; piles need the YOLO-seg stretch goal.
* "What can I build?" only knows our 50 part types. Use **"only parts our model knows"**
  (on by default) — the app shows how much of each set that covers (`recognisable`).
* Similar colours (light/dark bluish grey, black/dark blue under weak light) need the
  measured palette; unsure bricks are flagged and lower the set score.
