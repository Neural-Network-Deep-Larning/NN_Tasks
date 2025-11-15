# data.py
import streamlit as st
import pandas as pd
from typing import Tuple

@st.cache_data
def load_penguins(path: str = None) -> pd.DataFrame:
    """
    Loads penguins CSV. If path is None, tries local 'penguins.csv' in project root.
    """
    if path is None:
        path = "penguins.csv"
    df = pd.read_csv(path)
    # rename columns to match earlier code
    df = df.rename(columns={
        "species": "Species",
        "culmen_length_mm": "CulmenLength",
        "culmen_depth_mm": "CulmenDepth",
        "flipper_length_mm": "FlipperLength",
        "body_mass_g": "BodyMass",
        "island": "OriginLocation"
    })
    df = df[["Species", "CulmenLength", "CulmenDepth", "FlipperLength", "OriginLocation", "BodyMass"]]
    return df
