"""
Random Forest finale con parametri ottimali
+ Leave-One-Subject-Out (GroupKFold)
+ stampa Accuracy, ROC-AUC, Log-loss medi
+ salva il modello addestrato su TUTTO il dataset
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.compose      import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.ensemble      import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             log_loss, make_scorer)

#  carica il CSV 
csv = Path(r"C:\Users\bombe\OneDrive\Desktop\DatiStress\DataSet1"
           r"\physiological_signals_30sn_reduced_no2.csv")
df  = pd.read_csv(csv)

feature_cols = ["eda", "bvp", "temp"]
X      = df[feature_cols]
y      = df["emotion"]
groups = df["subjet"]

# scaler 
use_scaler = False  
if use_scaler:
    preproc = ColumnTransformer(
        [("num", StandardScaler(), feature_cols)],
        remainder="passthrough"
    )
else:
    preproc = "passthrough"

#  modello con i parametri migliori 
rf = RandomForestClassifier(
    n_estimators      = 400,
    max_depth         = None,
    max_features      = "sqrt",
    min_samples_leaf  = 1,
    class_weight      = "balanced",
    random_state      = 42,
    n_jobs            = -1
)

pipe = Pipeline([
    ("prep", preproc),
    ("rf",   rf)
])

#   Leave-One-Subject-Out CV 
gkf = GroupKFold(n_splits=groups.nunique())

acc  = cross_val_score(pipe, X, y,
                       cv=gkf, groups=groups,
                       scoring="accuracy", n_jobs=-1)

auc  = cross_val_score(pipe, X, y,
                       cv=gkf, groups=groups,
                       scoring=make_scorer(roc_auc_score,
                                           needs_proba=True,
                                           labels=[0,1]),
                       n_jobs=-1)

logl = -cross_val_score(pipe, X, y,
                        cv=gkf, groups=groups,
                        scoring=make_scorer(log_loss,
                                            needs_proba=True,
                                            greater_is_better=False,
                                            labels=[0,1]),
                        n_jobs=-1)

print(f"Accuracy media : {acc.mean():.3f}")
print(f"ROC-AUC  media : {auc.mean():.3f}")
print(f"Log-loss media : {logl.mean():.4f}")

#  addestra su TUTTO il dataset e salva 
pipe.fit(X, y)                       # un solo training finale
joblib.dump(pipe, "rf_best.pkl")     # modello + scaler 
print("Modello salvato in rf_best.pkl")


