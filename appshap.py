import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Charger le modèle
model = joblib.load("models/model.pkl")

# Extraire le modèle final (RandomForest) depuis le pipeline
regressor = model.named_steps["regressor"]
preprocessor = model.named_steps["preprocessing"]

# Expliquer une instance
def explain_prediction(input_dict):
    df_input = pd.DataFrame([input_dict])
    X_transformed = preprocessor.transform(df_input).astype("float64")

    explainer = shap.Explainer(regressor)
    shap_values = explainer(X_transformed, check_additivity=False)

    print("🔍 Contribution de chaque variable :")
    for feature, shap_val in zip(df_input.columns, shap_values.values[0]):
        print(f"{feature}: {shap_val:.2f}")

    shap.plots.waterfall(shap_values[0])
    plt.show()



if __name__ == "__main__":
    sample_input = {
        "Country": "France",
        "Employment": "Employed full-time",
        "RemoteWork": "Remote",
        "EdLevel": "Master’s degree",
        "DevType": "Full-stack developer",
        "YearsCodePro": 5
    }

    explain_prediction(sample_input)
