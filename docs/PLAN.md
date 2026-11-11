# Work plan

Roles follow the kickoff talk; everyone reviews everyone's pull requests.

| Person | Owns | First deliverable |
|---|---|---|
| Person 1 — data & tools | Kaggle setup, data folders, photo sessions, W&B, CVAT | renders load on Kaggle, first 200 real crops |
| Person 2 — colour | grey card, `colour.py`, colour calibration, colour eval | measured palette + leave-one-photo-out accuracy |
| Person 3 — shape | `segment.py`, CNN + baseline, core experiment, Grad-CAM | experiment table A/B/C + baseline |
| Person 4 — goal & app | Rebrickable, `sets.py`, Streamlit app, demo | app runs end-to-end on one real photo |

## Timeline (suggestion — adjust to your deadlines)

| Week | Goal | Done when |
|---|---|---|
| 1 | Setup | repo on GitHub, task board, Kaggle notebook trains A on 2 epochs, Rebrickable prepared |
| 2 | Capture rig + first photos | fixed camera, 2 LEDs, grey card; `extract_crops.py` works on 5 classes |
| 3 | Baseline + run A | first numbers on real crops — **this is the gap we must close** |
| 4 | Photos for all 50 classes | ≥ 10 photos per class (≥ 60 crops) ; colour photos for 15 colours |
| 5 | Runs B, C (+ D if pretrained allowed) | experiment table filled; confusions analysed |
| 6 | Colour calibrated, app end-to-end | inventory of mixed photos vs. hand count |
| 7 | Stretch: piles (YOLO-seg) or more classes | only if 1–6 are solid |
| 8 | Report (Overleaf) + demo | figures from `results/` |

## Photo protocol (the most important practical part)

* Camera on a stand, **same height and angle every time**; phone on fixed exposure/white balance if possible.
* Two LED panels from both sides, about 45° → fewer shadows.
* Matte, plain background. Pick a colour **not** in your brick set (e.g. a light, slightly
  bluish card if you have no light-blue bricks). Avoid white — white bricks vanish.
* Grey card always in the **top-left corner** (`grey_card_roi` in the config).
* Bricks **not touching**, ≥ 1 cm apart.
* **Class photos** (for training/test): one part type per photo, 5–10 bricks, mixed colours,
  mixed poses — top, side, upside down. Folder name = class name.
* **Mixed photos** (for the inventory demo): several types; count by hand → ground truth.
* **Colour photos**: one colour per photo, several part types.
* Test set: photos taken **on a different day** (or different lamp position) than train — the honest test.

## Core experiment (slide 5): renders → real

All rows evaluated on the same `data/real/test`:

| Run | Training data | Augmentation | Pretrained | Acc | Top-3 | Macro-F1 |
|---|---|---|---|---|---|---|
| Baseline | renders (+ real) | real | – | | | |
| A | renders | basic | no | | | |
| B | renders | real | no | | | |
| C | renders → fine-tune real | real | no | | | |
| D | as C | real | yes (if allowed) | | | |
| E | real only | basic | no | | | |

Extra analysis for the report:
* **Learning curve**: C with 10 / 25 / 50 / 100 % of the real crops — how many own photos do we need?
* Most-confused pairs (`confusions.csv`) — expect plate vs tile, 1×2 vs 1×3, slopes.
* Grad-CAM: does the network look at the studs?
* Calibration: confidence when right vs wrong (`metrics.json`) — justifies the "unsure" flag.

## End-to-end evaluation

On 10–20 mixed photos with hand-counted ground truth:
* detection: found / missed / false bricks
* inventory: exact-match rate of (part, colour, count) lines
* time per photo
