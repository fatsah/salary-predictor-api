import pandas as pd
from pandasql import sqldf

# 📌 Charger les données
df = pd.read_csv("data/survey_results_public.csv")

# 📌 Requête SQL sur le DataFrame
query = """
SELECT *
FROM df
--WHERE Country = 'France'
  --AND Employment IS NOT NULL
  --AND DevType IS NOT NULL
LIMIT 2
"""
result = sqldf(query)
for col in result.columns:
    print(col)

print(result.head())

# ✅ Fonction d'affichage avec colonnes à largeur fixe
def afficher_tableau_fixe(df, cols, col_widths=None, max_lignes=10):
    def format_cell(val, width):
        val_str = str(val)
        return val_str[:width-3] + "..." if len(val_str) > width else val_str.ljust(width)

    if col_widths is None:
        col_widths = {col: 25 for col in cols}

    header = "│ " + " │ ".join([col.ljust(col_widths[col]) for col in cols]) + " │"
    ligne_sep = "├" + "─" * (len(header) - 2) + "┤"
    bord_haut = "┌" + "─" * (len(header) - 2) + "┐"
    bord_bas = "└" + "─" * (len(header) - 2) + "┘"

    print(bord_haut)
    print(header)
    print(ligne_sep)

    for _, row in df.head(max_lignes).iterrows():
        ligne = "│ " + " │ ".join([format_cell(row[col], col_widths[col]) for col in cols]) + " │"
        print(ligne)

    print(bord_bas)

# 📌 Requête SQL sur le DataFrame
query_filtred = """
SELECT Country, Employment, CompTotal, ConvertedCompYearly, RemoteWork, EdLevel, Age, YearsCodePro
FROM df
WHERE upper(Country) like upper('%luxembourg%')
  AND CompTotal IS NOT NULL
  AND EdLevel IS NOT NULL
LIMIT 2
"""
result_filtred = sqldf(query_filtred)

# ✅ Colonnes à afficher + largeur personnalisée
colonnes = ["Country", "Employment", "CompTotal", "ConvertedCompYearly", "RemoteWork", "EdLevel", "Age", "YearsCodePro"]
largeurs = {
    "Country": 15,
    "Employment": 30,
    "CompTotal": 8,
    "ConvertedCompYearly": 8,
    "RemoteWork": 10,
    "EdLevel": 25,
    "Age": 15,
    "YearsCodePro": 3
}

# 🖨️ Affichage
afficher_tableau_fixe(result_filtred, colonnes, largeurs)
