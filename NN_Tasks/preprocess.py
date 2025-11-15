# preprocess.py
from typing import Tuple, Dict
import pandas as pd
import numpy as np

def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    # fill numeric NaNs per-species mean
    for col in ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]:
        df[col] = df.groupby('Species')[col].transform(lambda x: x.fillna(x.mean()))
    # origin
    if df["OriginLocation"].isnull().any():
        df["OriginLocation"] = df["OriginLocation"].fillna(df["OriginLocation"].mode()[0])
    orig_levels = sorted(df["OriginLocation"].unique().tolist())
    origin_map = {v: i for i, v in enumerate(orig_levels)}
    df["OriginLocation_enc"] = df["OriginLocation"].map(origin_map)
    info = {"origin_map": origin_map, "orig_levels": orig_levels}
    return df, info

def split_by_class(data: pd.DataFrame, target_col: str, chosen_classes: tuple, seed: int = 42):
    import numpy as np
    np.random.seed(seed)
    train_list, test_list = [], []
    for c in chosen_classes:
        class_rows = data[data[target_col] == c].copy()
        total = len(class_rows)
        shuffled = np.random.permutation(class_rows.index)
        cutoff = 30 if total >= 50 else int(total * 0.6)
        train_rows = class_rows.loc[shuffled[:cutoff]]
        test_rows = class_rows.loc[shuffled[cutoff:]]
        train_list.append(train_rows)
        test_list.append(test_rows)
    train_data = pd.concat(train_list, ignore_index=True)
    test_data = pd.concat(test_list, ignore_index=True)
    return train_data, test_data
