# Résumé du Module 02 : Experiment Tracking & Model Management

## 1. Vue d'ensemble & Objectifs

Le module 02 aborde la transition entre le prototypage ponctuel et l'ingénierie rigoureuse de modèles via **MLflow**. Il résout le problème de la perte d'historique des expérimentations (quels hyperparamètres ont donné quel score ? quelles données ont été utilisées ? où est le fichier du modèle ?).

### Objectifs Principaux
1. **Experiment Tracking** : Journaliser et comparer systématiquement les runs, paramètres, métriques, artefacts et dépendances.
2. **Model Management & Model Registry** : Centraliser les modèles entraînés, gérer les versions, assurer la traçabilité (*lineage*) et organiser les étapes de validation (Staging / Production ou Champion / Challenger).
3. **Comprendre les topologies de déploiement** : Du serveur local simple à l'architecture d'équipe dans le Cloud.

### Les 3 Scénarios d'Architecture MLflow
- **Scénario 1 (Local basique)** : Un seul utilisateur, pas de base de données dédiée, métadonnées et artefacts stockés dans le système de fichiers local (`./mlruns`).
- **Scénario 2 (Local avec backend SQLite)** : Métadonnées stockées dans SQLite (`sqlite:///mlflow.db`) et artefacts en local (`./artifacts` ou `./mlruns`). Ce mode débloque les fonctionnalités complètes du **Model Registry**.
- **Scénario 3 (Architecture partagée / Cloud)** : Déploiement sur AWS (ou GCP/Azure) avec :
  - **Serveur de tracking MLflow** hébergé sur une instance EC2.
  - **Backend Store** sur une base managée PostgreSQL (AWS RDS).
  - **Artifact Store** sur un bucket objet S3 (`s3://<bucket-name>`).

---

## 2. Ce qu'il faut absolument retenir (Points clés)

1. **Les 4 éléments essentiels journalisés par run MLflow** :
   - **Paramètres** (`mlflow.log_param`, `mlflow.log_params`) : Hyperparamètres, chemins des fichiers de données (`train_data_path`, etc.).
   - **Métriques** (`mlflow.log_metric`, `mlflow.log_metrics`) : RMSE, score d'évaluation calculé sur chaque run.
   - **Tags** (`mlflow.set_tag`, `mlflow.set_tags`) : Auteur, environnement, type de modèle, etc.
   - **Artefacts & Modèle** (`mlflow.log_artifact`, `mlflow.xgboost.log_model`, `mlflow.sklearn.log_model`) : Modèle sérialisé, graphiques, environnement Conda/Pip généré automatiquement.
2. **Ne jamais oublier le préprocesseur (Lineage complet)** :
   - Sauvegarder le modèle seul est insuffisant en production si la structure du `DictVectorizer` est perdue.
   - Toujours logger l'artefact de préprocessing : `mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")`.
3. **Autologging vs Logging Explicite** :
   - `mlflow.xgboost.autolog()` ou `mlflow.sklearn.autolog()` capturent automatiquement les paramètres et métriques usuels.
   - **Attention** : L'autologging ne sait pas toujours quels artefacts annexes (comme le préprocesseur) ou chemins de données métier doivent être conservés. Il est souvent préférable de combiner l'autologging avec du logging explicite d'artefacts.
4. **Optimisation d'Hyperparamètres avec Hyperopt** :
   - Définir un espace de recherche (`hp.choice`, `hp.uniform`, `hp.loguniform`, `hp.quniform`).
   - Encapsuler la fonction objectif dans `with mlflow.start_run():` pour que chaque évaluation de Hyperopt génère un run individuel consultable et comparable dans l'UI MLflow.
5. **Model Registry & Évolution des Stages** :
   - Un modèle dans le Registry est versionné (`v1`, `v2`, etc.).
   - **Important (MLflow 2.9+)** : Les *stages* historiques (`Staging`, `Production`, `Archived`) sont dépréciés. La bonne pratique moderne repose sur les **Aliases** (ex: `@champion`, `@challenger`, `@bad`) et les **Tags** (`client.set_registered_model_alias(...)`).

---

## 3. Modèles Utilisés

| Modèle | Librairie / Framework | Contexte & Spécificités |
| :--- | :--- | :--- |
| **XGBoost (Booster)** | `xgboost` (`xgb.train`, `DMatrix`) | Modèle principal pour le tuning fin. Hyperparamètres explorés : `max_depth`, `learning_rate`, `min_child_weight`, `objective='reg:squarederror'`. |
| **Lasso** | `sklearn.linear_model.Lasso` | Utilisé pour illustrer le logging manuel simple avec variation d'`alpha`. |
| **RandomForestRegressor** | `sklearn.ensemble.RandomForestRegressor` | Testé lors de la comparaison systématique avec `mlflow.sklearn.autolog()`. |
| **GradientBoostingRegressor** | `sklearn.ensemble.GradientBoostingRegressor` | Arbre de boosting Scikit-Learn évalué en multi-modèles. |
| **ExtraTreesRegressor** | `sklearn.ensemble.ExtraTreesRegressor` | Forêt d'arbres extrêmement aléatoires. |
| **LinearSVR** | `sklearn.svm.LinearSVR` | Support Vector Machine linéaire pour la régression. |

---

## 4. Commandes et Extraits Importants

### Commandes CLI / Shell

```bash
# 1. Lancement de l'UI MLflow locale avec base SQLite (active le Model Registry)
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 2. Lancement du serveur MLflow distant (ex: VM AWS EC2 vers RDS et S3)
mlflow server \
    -h 0.0.0.0 \
    -p 5000 \
    --backend-store-uri postgresql://DB_USER:DB_PASS@DB_ENDPOINT:5432/DB_NAME \
    --default-artifact-root s3://MON_BUCKET_ARTIFACTS

# 3. Installation des dépendances du module
pip install mlflow hyperopt xgboost boto3 psycopg2-binary
```

### Extraits Python Clés

#### A. Initialisation et Logging Manuel avec Hyperopt & XGBoost

```python
import mlflow
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

# Configuration de l'URI et de l'expérience
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

# Définition de l'espace de recherche
search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:squarederror',
    'seed': 42
}

def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=100,
            evals=[(dval, 'validation')],
            early_stopping_rounds=10
        )
        
        y_pred = booster.predict(dval)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        
        return {'loss': rmse, 'status': STATUS_OK}

best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=20,
    trials=Trials()
)
```

#### B. Sauvegarde du Modèle Final & du Préprocesseur

```python
with mlflow.start_run():
    mlflow.log_params(best_params)
    booster = xgb.train(params=best_params, dtrain=dtrain, num_boost_round=100)
    
    # Logging de la métrique
    mlflow.log_metric("rmse", rmse)
    
    # Sauvegarde du DictVectorizer
    mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
    
    # Logging du modèle XGBoost au format standard MLflow
    mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
```

#### C. Utilisation de MlflowClient & Model Registry

```python
from mlflow.tracking import MlflowClient
import mlflow

client = MlflowClient("sqlite:///mlflow.db")

# 1. Enregistrement d'un modèle à partir d'un run
run_id = "<RUN_ID>"
model_uri = f"runs:/{run_id}/models_mlflow"
mlflow.register_model(model_uri=model_uri, name="nyc-taxi-regressor")

# 2. Gestion moderne via Aliases (remplace les stages dépréciés)
client.set_registered_model_alias(
    name="nyc-taxi-regressor",
    alias="champion",
    version="1"
)

# 3. Chargement et inférence du modèle enregistré
model = mlflow.pyfunc.load_model("models:/nyc-taxi-regressor@champion")
predictions = model.predict(X_val)
```
