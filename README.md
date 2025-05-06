data/physiological_signals_30sn_reduced_no2.csv
Il dataset già filtrato (solo finestre di 30 s), pronto per il training.

src/Train_Model.py
Lo script principale che:

Carica il CSV da data/

Esegue Leave-One-Subject-Out CV e stampa le metriche (Accuracy, ROC-AUC, Log-loss)

Addestra la Random Forest sul dataset intero

Salva il modello finale in model/rf_best.pkl

src/New_Data_Predictions.py
Un piccolo script “in potenza” per eventuali nuove predizioni:
carica rf_best.pkl, legge un CSV con colonne eda,bvp,temp e genera un file con probabilità di stress.


requirements.txt
Qui dentro ci sono le versioni esatte di Python e librerie usate (pandas, scikit-learn, joblib…).

.gitignore
Per escludere venv, file IDE, dati grezzi o modelli troppo grandi.

2. Setup dell’ambiente
Clona il repo:

git clone https://github.com/AndreaCarlini01/Stress_analysis_And_Prediction.git
cd Stress_analysis_And_Prediction
Crea un virtualenv Python (3.10+):


python -m venv venv
Attiva il virtualenv:

venv\Scripts\activate
macOS/Linux

source venv/bin/activate
Installa le librerie:

pip install -r requirements.txt
3. Come addestrare il modello
Tutto il lavoro di training è già pronto in src/Train_Model.py.
Basta far partire:

python src/Train_Model.py
Cosa succede:

Viene caricato data/physiological_signals_30sn_reduced_no2.csv

Si esegue una validazione Leave-One-Subject-Out:

stampa Accuracy media, ROC-AUC media, Log-loss media

Poi la Random Forest viene rifittata su tutti i dati

Il risultato, modello + preprocessing, viene salvato in
model/rf_best.pkl
