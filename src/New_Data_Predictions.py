import joblib
import pandas as pd

#  carica il modello già addestrato
MODEL_PATH = r"../model/rf_best.pkl"  # cartella model
model = joblib.load(MODEL_PATH)

#  carica i nuovi dati
CSV_IN  = r"nuove_finestre.csv"            #  colonne: eda,bvp,temp
df_new  = pd.read_csv(CSV_IN)

#  estrai le stesse feature usate in training
X_new = df_new[["eda", "bvp", "temp"]]

#  predici
proba = model.predict_proba(X_new)[:, 1]   # probabilità stress (classe 1)
label = model.predict(X_new)               # etichetta 0 / 1

#  aggiungi al dataframe e salva
df_new["prob_stress"] = proba
df_new["pred_label"]  = label

CSV_OUT = r"nuove_finestre_pred.csv"
df_new.to_csv(CSV_OUT, index=False)
print(f"✓ Predizioni salvate in {CSV_OUT}")