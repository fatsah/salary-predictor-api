import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from src.data_preprocessing import load_and_preprocess, build_transformer
import os

# Création du dossier si besoin
os.makedirs("models", exist_ok=True)


def train():
    X, y = load_and_preprocess("data/cleaned_survey.csv")
    preprocessor = build_transformer(X)

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)
    joblib.dump(model, "models/model.pkl")
    print("✅ Modèle entraîné et sauvegardé dans models/model.pkl")

if __name__ == "__main__":
    train()
