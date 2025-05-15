import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["ConvertedCompYearly", "YearsCodePro"])
    df["YearsCodePro"] = pd.to_numeric(df["YearsCodePro"], errors='coerce')
    df = df[df["YearsCodePro"].notnull()]
    y = df["ConvertedCompYearly"]
    X = df.drop(columns=["ConvertedCompYearly"])
    return X, y

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_transformer(X):
    num_features = ["YearsCodePro"]
    cat_features = ["Country", "Employment", "RemoteWork", "EdLevel", "DevType"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # ✅ dense output
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)  # ✅ ne PAS mettre OneHotEncoder ici !
    ])

    return preprocessor

