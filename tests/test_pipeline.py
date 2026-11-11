"""Run with:  python -m pytest -q   (or just  python tests/test_pipeline.py)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from lego import sets as S  # noqa: E402
from lego.capture import white_balance  # noqa: E402
from lego.config import load_config  # noqa: E402
from lego.pipeline import Pipeline  # noqa: E402
from synthetic import BRICKS, COLOURS, make_photo, write_fake_rebrickable  # noqa: E402

NAMES = {4: "Red", 1: "Blue", 14: "Yellow", 0: "Black", 2: "Green"}


class FakeShape:
    """Pretends to be the CNN: answers by brick size so we can test the plumbing."""

    def predict(self, greys, topk=3):
        return [[("3001 brick 2x4", 0.9), ("3003 brick 2x2", 0.05)] for _ in greys]


def test_white_balance_removes_cast():
    img = make_photo()
    out, gains = white_balance(img, [0.02, 0.02, 0.14, 0.14])
    card = out[30:140, 50:200].reshape(-1, 3).mean(0)
    assert np.ptp(card) < 4, card  # neutral after correction


def test_pipeline_finds_bricks_and_colours():
    pipe = Pipeline(load_config(), shape_model=FakeShape())
    res = pipe.run(make_photo())
    assert len(res.detections) == len(BRICKS), res.detections_df
    # match each true brick to the nearest detection centre
    got = {}
    for d in res.detections:
        cx, cy = (d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2
        j = int(np.argmin([np.hypot(cx - b[0], cy - b[1]) for b in BRICKS]))
        got[j] = d.colour_name
    wrong = [(NAMES[BRICKS[j][4]], c) for j, c in got.items() if NAMES[BRICKS[j][4]] != c]
    assert not wrong, wrong
    assert res.inventory["count"].sum() == len(BRICKS)


def test_set_matching(tmp_path):
    write_fake_rebrickable(tmp_path)
    sp, sets = S.build_set_parts(tmp_path)
    assert "9999" not in set(sp["part"])  # old inventory version ignored
    inv = pd.DataFrame({
        "part": ["3001", "3003", "3004"], "colour_id": [4, 1, 14],
        "count": [2, 2, 2], "expected": [2.0, 1.5, 2.0],
    })
    res = S.what_can_i_build(inv, sp, sets, weighted=False, min_parts=0)
    top = res.iloc[0]
    assert top["set_num"] == "1-1" and top["coverage"] == 1.0, res
    res_w = S.what_can_i_build(inv, sp, sets, weighted=True, min_parts=0)
    assert res_w.set_index("set_num").loc["1-1", "coverage"] < 1.0  # unsure bricks lower the score
    miss = S.missing_parts("2-1", inv, sp, weighted=False)
    assert set(miss["part"]) == {"3001", "3069b"}


if __name__ == "__main__":
    import tempfile

    test_white_balance_removes_cast()
    test_pipeline_finds_bricks_and_colours()
    with tempfile.TemporaryDirectory() as t:
        test_set_matching(Path(t))
    print("all tests passed")
