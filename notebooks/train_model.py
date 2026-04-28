import pandas as pd
import numpy as np

# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")

# Verifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")
from sklearn.preprocessing import LabelEncoder

# Encoder les variables categoriques
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Definir les features (X) et la cible (y)
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")
print(f"Cible : {y.shape}")

from sklearn.model_selection import train_test_split

# 80% entrainement, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Entrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")
from sklearn.ensemble import RandomForestClassifier

# Creer le modele
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Entrainer
model.fit(X_train, y_train)

print("Modele entraine !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Prédire sur les données de test
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Matrice de confusion :")
print(cm)

# Rapport détaillé
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

import joblib
import os

# Créer le dossier models/ s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Sauvegarder le modèle
joblib.dump(model, "models/model.pkl")

# Sauvegarder les encodeurs
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")

# Sauvegarder la liste des features
joblib.dump(feature_cols, "models/feature_cols.pkl")

# Vérifier la taille du fichier
size = os.path.getsize("models/model.pkl")
print(f"Modèle sauvegardé : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")
print("Encodeurs et metadata sauvegardés.")

# Recharger le modèle depuis le fichier
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"Modèle rechargé : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# Nouveau patient fictif
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

# Encoder
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# Vecteur de features
features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

# Prédire
diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

print(f"\n--- Résultat du pré-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilité : {proba_max:.1%}")

# Exercice 1 - Importance des features
importances = model.feature_importances_
print("\n--- Importance des features ---")
for name, imp in sorted(zip(feature_cols, importances),
                        key=lambda x: x[1], reverse=True):
    print(f"{name:20s} : {imp:.3f}")

    # Exercice 2 - 3 patients fictifs
patients_test = [
    {'age': 16, 'sexe': 'M', 'temperature': 36.8, 'tension_sys': 120,
     'toux': False, 'fatigue': False, 'maux_tete': False, 'region': 'Dakar'},
    {'age': 35, 'sexe': 'F', 'temperature': 40.2, 'tension_sys': 105,
     'toux': False, 'fatigue': True, 'maux_tete': True, 'region': 'Thiès'},
    {'age': 68, 'sexe': 'M', 'temperature': 38.9, 'tension_sys': 145,
     'toux': True, 'fatigue': True, 'maux_tete': False, 'region': 'Saint-Louis'},
]

print("\n--- Exercice 2 : 3 patients fictifs ---")
for i, p in enumerate(patients_test):
    sexe_enc = le_sexe_loaded.transform([p['sexe']])[0]
    region_enc = le_region_loaded.transform([p['region']])[0]
    features = [p['age'], sexe_enc, p['temperature'], p['tension_sys'],
                int(p['toux']), int(p['fatigue']), int(p['maux_tete']), region_enc]
    diag = model_loaded.predict([features])[0]
    proba = model_loaded.predict_proba([features])[0].max()
    print(f"Patient {i+1} ({p['sexe']}, {p['age']} ans, {p['temperature']}°C) : {diag} ({proba:.1%})")
    