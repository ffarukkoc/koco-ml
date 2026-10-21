"""Small reproducible tables shared by the tests."""

import numpy as np
import pandas as pd

def RegressionTable(Rows: int = 300, Seed: int = 0) -> pd.DataFrame:
    Rng = np.random.RandomState(Seed)
    Table = pd.DataFrame({
        "a": Rng.normal(size=Rows),
        "b": Rng.normal(size=Rows),
        "c": Rng.normal(size=Rows),
    })
    Table["y"] = 2 * Table["a"] + Table["b"] + Rng.normal(size=Rows) * 0.5
    return Table

def BinaryTable(Rows: int = 300, Seed: int = 0) -> pd.DataFrame:
    Table = RegressionTable(Rows, Seed)
    Table["y"] = (Table["y"] > 0).astype(int)
    return Table

def NoiseTable(Features: int = 10, Rows: int = 300, Seed: int = 0) -> pd.DataFrame:
    Rng = np.random.RandomState(Seed)
    Table = pd.DataFrame(Rng.normal(size=(Rows, Features)), columns=[f"f{i}" for i in range(Features)])
    Table["y"] = Rng.normal(size=Rows)
    return Table

def WOETable(Rows: int = 3000, Seed: int = 0) -> pd.DataFrame:
    Rng = np.random.RandomState(Seed)
    Table = pd.DataFrame({
        "income": Rng.lognormal(8, 1, Rows),
        "age": Rng.randint(18, 80, Rows).astype(float),
        "score": Rng.normal(0, 1, Rows),
        "noise": Rng.normal(0, 1, Rows),
        "sparse": np.where(Rng.rand(Rows) < 0.3, np.nan, Rng.normal(size=Rows)),
        "tiny_null": np.where(Rng.rand(Rows) < 0.004, np.nan, Rng.normal(size=Rows)),
        "const": 1.0,
    })
    Logit = -2 + 0.9 * Table["score"] + 0.00003 * Table["income"] - 0.02 * (Table["age"] - 40) + 0.5 * Table["sparse"].fillna(0)
    Table["y"] = (Rng.rand(Rows) < 1 / (1 + np.exp(-Logit))).astype(int)
    return Table
