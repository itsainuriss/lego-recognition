"""LEGO brick recognition: photo in, brick list out, then 'What can I build?'.

Pipeline (one module per step):
    capture.py    1  load photo, white balance with the grey card
    segment.py    2  find every brick and cut it out
    shape_*.py    3a which part is it?   (CNN, or HOG+SVM baseline)
    colour.py     3b which colour is it? (measure + nearest reference, no NN)
    inventory.py  4  part, colour, count + how sure we are
    sets.py       5  "What can I build?" against Rebrickable part lists
    pipeline.py      glues the steps together
"""

__version__ = "0.1.0"
