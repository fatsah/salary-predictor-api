from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
from src.data_preprocessing import load_and_preprocess

def evaluate():
    # Charger les données et le modèle
    X, y = load_and_preprocess("data/cleaned_survey.csv")
    model = joblib.load("models/model.pkl")

    # Prédictions
    y_pred = model.predict(X)

    # Évaluation
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    # Résultats
    print("📊 Évaluation du modèle :")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.2f}")

if __name__ == "__main__":
    evaluate()
