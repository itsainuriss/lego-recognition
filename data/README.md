# data/ (not in git)

```
data/
  renders/            Kaggle "Images of LEGO Bricks" (or use /kaggle/input/... directly)
  photos/             our class photos: one part type per photo, folder = class name
  real_all/           crops from scripts/extract_crops.py
  real/train, test/   split by photo with scripts/split_real.py
  colour_photos/      one colour per folder: "<rebrickable id> <name>"
  rebrickable/        sets / inventories / inventory_parts / colors .csv.gz + prepared.pkl
```

Folder names of `photos/` must start with the **same part number** as the
Kaggle class folders (e.g. `3001 brick 2x4`). The rest of the name does not matter.
