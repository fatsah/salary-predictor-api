import pandas as pd
import joblib

def predict(input_dict):
    model = joblib.load("models/model.pkl")
    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)
    return prediction[0]

# Exemple d’utilisation
if __name__ == "__main__":
    exemple = {
        "Country": "France",
        "Employment": "Employed full-time",
        "RemoteWork": "Remote",
        "EdLevel": "Master’s degree",
        "DevType": "Full-stack developer",
        "YearsCodePro": 5
    }
    print("Salaire prédit :", predict(exemple))
