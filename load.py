import pandas as pd


# Chargement
df = pd.read_csv("data/survey_results_public.csv", encoding='utf-8')
print("Dimensions initiales :", df.shape)

# Colonnes utiles
cols_of_interest = [
    "Country", "Employment", "RemoteWork", "EdLevel",
    "YearsCodePro", "DevType", "LanguageHaveWorkedWith",
    "CompTotal", "ConvertedCompYearly"
]

# Conserver uniquement ces colonnes (même si elles ont des NaN)
df_cleaned = df[cols_of_interest]

# Supprimer les lignes où la colonne salaire est manquante ou trop élevée
df_cleaned = df_cleaned[df_cleaned["ConvertedCompYearly"].notnull()]
df_cleaned = df_cleaned[df_cleaned["ConvertedCompYearly"] < 500000]

# Nettoyage des années d'expérience (valeurs non numériques à convertir)
df_cleaned["YearsCodePro"] = df_cleaned["YearsCodePro"].replace(
    {"Less than 1 year": 0.5, "More than 50 years": 50}
)
df_cleaned["YearsCodePro"] = pd.to_numeric(df_cleaned["YearsCodePro"], errors='coerce')
df_cleaned = df_cleaned[df_cleaned["YearsCodePro"].notnull()]

# Aperçu final
print("Dimensions après nettoyage :", df_cleaned.shape)
print(df_cleaned.head())

# Export
df_cleaned.to_csv("data/cleaned_survey.csv", index=False)
print("Colonnes finales :", df_cleaned.columns.tolist())
