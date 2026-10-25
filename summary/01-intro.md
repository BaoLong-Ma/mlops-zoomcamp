# Résumé du Module 01 : Introduction & Baseline Model

## 1. Vue d'ensemble & Objectifs

Le module 01 pose les bases de l'ingénierie MLOps à travers un cas d'usage concret : **la prédiction du temps de trajet (durée en minutes) des taxis new-yorkais (NYC Taxi dataset)**. 

L'objectif principal est de comprendre le cycle de vie d'un projet de Machine Learning, de passer d'un simple notebook exploratoire à un modèle baseline structuré, et de situer son projet dans l'échelle de maturité MLOps.

### Le Cycle de Vie ML (3 étapes clés)
1. **Design** : Comprendre le problème métier, vérifier si le ML est la solution adaptée, définir la métrique d'évaluation.
2. **Train** : Ingestion et nettoyage des données, ingénierie des caractéristiques (*feature engineering*), entraînement de modèles comparatifs, évaluation.
3. **Operate** : Déploiement du modèle (API, batch, streaming), monitoring de la dérive des données et des performances en continu.

### Le Modèle de Maturité MLOps
- **Niveau 0 (Aucune automatisation)** : Code 100% dans des notebooks, pas de tracking d'expériences, transition manuelle data science -> ingénierie. Convient aux PoC.
- **Niveau 1 (DevOps sans MLOps)** : Bonnes pratiques d'ingénierie logicielle (CI/CD, tests automatisés), mais absence de gestion et traçabilité spécifique au ML (données, modèles).
- **Niveau 2 (Entraînement automatisé)** : Entraînement reproductible sous forme de scripts (`python train.py`), gestion des paramètres, tracking des runs. Cible principale du cours.
- **Niveau 3 (Déploiement automatisé)** : Pipeline CI/CD incluant le packaging du modèle, tests A/B ou Canary, déploiement automatisé du service.
- **Niveau 4 (Automatisation MLOps complète)** : Rétroaction automatique basée sur le monitoring (détection de *data drift* ou dégradation de performance -> ré-entraînement automatique -> validation -> déploiement).

---

## 2. Ce qu'il faut absolument retenir (Points clés)

1. **Format Parquet vs CSV** :
   - Les données modernes de taxi sont distribuées en format `.parquet` (compressé, typé en colonnes).
   - Nécessite l'installation d'un moteur comme `pyarrow` ou `fastparquet` pour être exploité avec Pandas (`pd.read_parquet`).
2. **Préparation des Données & Outliers** :
   - **Calcul de la cible** : `duration = (dropoff_datetime - pickup_datetime).total_seconds() / 60`.
   - **Nettoyage indispensable** : Filtrer les valeurs aberrantes (par exemple ne conserver que les trajets compris entre 1 et 60 minutes : `(df.duration >= 1) & (df.duration <= 60)`). Sans ce filtrage, le modèle apprend sur du bruit extrême.
3. **Feature Engineering & Encodage** :
   - Les identifiants de zones (`PULocationID`, `DOLocationID`) sont numériques à la base, mais doivent impérativement être traités comme des **variables catégorielles** (cast en `str`).
   - Utilisation de `DictVectorizer` de scikit-learn pour encoder les dictionnaires de variables catégorielles (One-Hot Encoding automatique) et numériques.
   - **Combinaison de features** : Concaténer la zone de prise en charge et de dépose (`PULocationID + '_' + DOLocationID`) capture les spécificités des trajets et réduit significativement l'erreur (RMSE).
4. **Indissociabilité Modèle / Préprocesseur** :
   - Le modèle entraîné attend des matrices creuses encodées exactement avec les mêmes colonnes que lors de l'entraînement.
   - **Règle d'or** : Sauvegarder impérativement le `DictVectorizer` (`dv`) en même temps que le modèle de régression (`pickle.dump((dv, lr), f)`).

---

## 3. Modèles Utilisés

| Modèle | Librairie | Rôle & Paramètres notables |
| :--- | :--- | :--- |
| **Linear Regression** | `sklearn.linear_model.LinearRegression` | **Modèle baseline** servant de point de comparaison initial. Simple, rapide à entraîner et interprétable. |
| **Lasso Regression** | `sklearn.linear_model.Lasso` | Régularisation L1 (force certains coefficients à zéro, utile pour la sélection de features). Paramètre clé : `alpha` (ex: `alpha=0.01`). |
| **Ridge Regression** | `sklearn.linear_model.Ridge` | Régularisation L2 (pénalise les coefficients de forte amplitude sans les annuler). |

- **Métrique d'évaluation** : **RMSE** (*Root Mean Squared Error*), via `root_mean_squared_error` ou `mean_squared_error(..., squared=False)`. Elle pénalise fortement les grands écarts en minutes.

---

## 4. Commandes et Extraits Importants

### Commandes Shell / Système

```bash
# Téléchargement des données Parquet
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-01.parquet
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2021-02.parquet

# Sécurisation de la clé SSH pour VM Cloud (AWS EC2)
chmod 400 ~/.ssh/my-key-pair.pem
ssh -i ~/.ssh/my-key-pair.pem ubuntu@<IP_PUBLIQUE>

# Test de bon fonctionnement de Docker
docker run hello-world

# Installation des dépendances Python requises
pip install pandas pyarrow fastparquet scikit-learn seaborn matplotlib
```

### Extraits Python Clés

```python
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import root_mean_squared_error

# 1. Lecture et préparation des données
def read_dataframe(filename):
    df = pd.read_parquet(filename)
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    # Feature engineering : combinaison des lieux
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    return df

df_train = read_dataframe('yellow_tripdata_2021-01.parquet')
df_val = read_dataframe('yellow_tripdata_2021-02.parquet')

# 2. Vectorisation des features
categorical = ['PU_DO']
numerical = ['trip_distance']

dv = DictVectorizer()
train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)

y_train = df_train['duration'].values
y_val = df_val['duration'].values

# 3. Entraînement et calcul du RMSE
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)
rmse = root_mean_squared_error(y_val, y_pred)
print(f"Validation RMSE: {rmse:.2f}")

# 4. Sauvegarde conjointe du préprocesseur et du modèle
with open('models/lin_reg.bin', 'wb') as f_out:
    pickle.dump((dv, lr), f_out)
```
