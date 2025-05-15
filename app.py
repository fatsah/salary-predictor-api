from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
import uuid
import os

# Charger le modèle pipeline
model = joblib.load("models/model.pkl")
preprocessor = model.named_steps["preprocessing"]
regressor = model.named_steps["regressor"]

app = FastAPI(title="Predict Developer Salary API", description="Prédit un salaire + explique + suggère", version="1.0")

# Données attendues
class DeveloperProfile(BaseModel):
    Country: str
    Employment: str
    RemoteWork: str
    EdLevel: str
    DevType: str
    YearsCodePro: float

# Fonction de recommandation simple
def get_recommendations(top_features):
    suggestions = []
    for f in top_features:
        if f["feature"] == "RemoteWork":
            suggestions.append("Chercher un poste en télétravail")
        elif f["feature"] == "DevType":
            suggestions.append("Monter en compétence pour devenir développeur full-stack")
        elif f["feature"] == "Country":
            suggestions.append("Cibler une entreprise internationale avec un meilleur salaire")
        elif f["feature"] == "EdLevel":
            suggestions.append("Poursuivre ses études pour un diplôme plus élevé")
        elif f["feature"] == "Employment":
            suggestions.append("Passer à un emploi à temps plein si possible")
        if len(suggestions) == 3:
            break
    return suggestions

def get_feature_names(preprocessor):
    num_features = preprocessor.transformers_[0][2]
    cat_pipeline = preprocessor.transformers_[1][1]
    cat_features = cat_pipeline.named_steps["onehot"].get_feature_names_out(preprocessor.transformers_[1][2])
    return list(num_features) + list(cat_features)



@app.post("/predict")
def predict_salary(data: DeveloperProfile):
    df = pd.DataFrame([data.dict()])
    X_transformed = preprocessor.transform(df).astype("float64")
    prediction = model.predict(df)[0]

    # SHAP
    explainer = shap.Explainer(regressor)
    shap_values = explainer(X_transformed, check_additivity=False)

    # On récupère les top 3 features influentes
    shap_impacts = shap_values.values[0]
    feature_names = get_feature_names(preprocessor)
    top_indices = sorted(range(len(shap_impacts)), key=lambda i: abs(shap_impacts[i]), reverse=True)[:3]
    top_factors = [
        {"feature": feature_names[i], "impact": round(shap_impacts[i], 2)}
        for i in top_indices
    ]


    # Recommandations associées
    suggestions = get_recommendations(top_factors)

    return {
        "predicted_salary": round(prediction, 2),
        "top_factors": top_factors,
        "suggestions": suggestions
    }

@app.post("/explain-plot")
def explain_plot(data: DeveloperProfile):
    df = pd.DataFrame([data.dict()])
    X_transformed = preprocessor.transform(df).astype("float64")
    prediction = model.predict(df)[0]

    # SHAP
    explainer = shap.Explainer(regressor)
    shap_values = explainer(X_transformed, check_additivity=False)

    # Générer graphique waterfall
    fig = shap.plots.waterfall(shap_values[0], show=False)
    file_path = f"shap_plot_{uuid.uuid4().hex}.png"
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

    return FileResponse(file_path, media_type="image/png", filename="shap_explanation.png")