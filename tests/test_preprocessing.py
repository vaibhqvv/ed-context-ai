import numpy as np
from src.preprocessing.cleaner import clip_vital_ranges
import pandas as pd


def test_clip_vital_ranges():
    df = pd.DataFrame({"heart_rate": [10, 85, 350], "sbp": [20, 120, 300]})
    result = clip_vital_ranges(df)
    assert result["heart_rate"].max() <= 280
    assert result["heart_rate"].min() >= 20
    print("PASS: clip_vital_ranges")


def test_missingness_mask():
    from src.preprocessing.windower import aggregate_window

    empty = pd.DataFrame()
    feats, mask = aggregate_window(empty, empty)
    assert all(v == 0 for v in mask.values())
    print("PASS: missingness mask on empty window")
