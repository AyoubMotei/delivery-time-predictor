# Projet de Prédiction du Temps de Livraison

## Objectif du Projet

Ce projet vise à construire un modèle de Machine Learning fiable capable de prédire le temps de livraison d'une commande en minutes. L'objectif principal est de fournir une estimation précise pour améliorer la logistique, l'expérience client et l'efficacité opérationnelle, tout en minimisant l'Erreur Moyenne Absolue (MAE).

## Structure du Projet

Le projet est organisé en séparant la logique d'expérimentation (Notebook.ipynb) du code de production et des tests.
```
├── dataset.csv                # Jeu de données original
├── Notebook.ipynb             # Logique d'expérimentation 
├── pipeline.py                # Code de production (Nettoyage, Prétraitement, Modélisation)
├── test_pipeline.py           # Tests unitaires pour garantir la qualité du code
└── README.md                  # Documentation du projet (Ce fichier)
```

## Démarche et Méthodologie

Le pipeline de production (`pipeline.py`) exécute les étapes suivantes de manière séquentielle et automatisée :

### 1. Préparation des Données

**Chargement et Nettoyage :**
- Les valeurs manquantes (NaN) dans la colonne d'expérience du coursier sont imputées par la médiane du jeu de données d'entraînement.
- Les valeurs manquantes dans les colonnes catégorielles (`Weather`, `Traffic_Level`, `Time_of_Day`) sont imputées par le mode (valeur la plus fréquente).

**Séparation :** Les données sont divisées en ensembles d'entraînement et de test (80/20).

### 2. Prétraitement (ColumnTransformer)

Pour préparer les données à la modélisation et garantir l'équité des échelles :

- **StandardScaler :** Appliqué aux variables numériques (`Distance_km`, `Courier_Experience_yrs`, `Preparation_Time_min`).
- **OneHotEncoder (OHE) :** Appliqué aux variables catégorielles (`Weather`, `Traffic_Level`, `Time_of_Day`, `Vehicle_Type`) pour les convertir en format numérique binaire.

### 3. Modélisation et Optimisation

Deux modèles de régression ont été testés et optimisés via `GridSearchCV` à l'intérieur d'un Pipeline :

- **Random Forest Regressor (RFR)**
- **Support Vector Regressor (SVR)**

**Validation Croisée :** `cv=5` (5-fold) est utilisé pour garantir la robustesse de l'optimisation.

**Métrique Cible :** L'optimisation utilise le `scoring='neg_mean_absolute_error'` pour maximiser la performance en MAE (Erreur Moyenne Absolue).

**Sélection de Features (Bonus) :** L'hyperparamètre `k` de `SelectKBest` (`score_func=f_regression`) a été inclus dans la grille d'hyperparamètres afin de déterminer si la réduction de la dimensionnalité améliorait la performance.

## Résultats et Modèle Adopté

L'exécution du pipeline a permis de comparer les deux modèles optimisés sur l'ensemble de test :

| Modèle | MAE (Validation Croisée) | MAE (Jeu de Test) | R² (Jeu de Test) |
|--------|--------------------------|-------------------|------------------|
| Random Forest (RFR) | 7.97 minutes | 6.87 minutes | 0.7910 |
| Support Vector Regressor (SVR) | 6.55 minutes | 5.79 minutes | 0.8230 |

### Justification du Choix

Le modèle **Support Vector Regressor (SVR)** a été adopté.

- **Précision :** Avec une MAE de **5.79 minutes** sur l'ensemble de test, le SVR surpasse le RFR d'environ 1.08 minute. En moyenne, la prédiction du temps de livraison ne s'écartera de la réalité que de moins de six minutes.
- **Explicabilité :** Le SVR obtient également un meilleur coefficient de détermination (R² = 0.8230), signifiant qu'il explique environ 82% de la variance observée dans le temps de livraison.
- **Hyperparamètres finaux du SVR :** `{'regressor__C': 1, 'regressor__kernel': 'linear'}` (Meilleure performance obtenue lorsque l'intégralité des features transformées est utilisée, `feature_selection__k: 'all'`).

## Tests Automatisés

Le fichier `test_pipeline.py` utilise `pytest` pour garantir la qualité du code de production. Deux types de tests cruciaux sont effectués :

### `test_format_dimension`
- Vérifie que le DataFrame n'est pas vide et que les ensembles d'entraînement/test ont la bonne taille.
- Vérifie l'intégrité du prétraitement : Confirme qu'aucune valeur manquante (NaN) ne subsiste après le `ColumnTransformer`, condition essentielle pour la modélisation.

### `test_seuil_mae`
- Vérifie l'exigence métier : Confirme que la MAE du modèle adopté (SVR) est inférieure au seuil maximal de 8.0 minutes (`MAE_SEUIL_MAX = 8.0`), assurant ainsi que le modèle reste performant en production.

**Résultat des tests :** Tous les tests ont passé avec succès (2 passed), validant la robustesse du pipeline.

## Comment Exécuter le Projet

Suivez ces étapes pour cloner le projet, installer les dépendances et exécuter le pipeline :

### 1. Cloner le Repository et Initialisation
```bash
# Cloner le repository
git clone https://github.com/AyoubMotei/delivery-time-predictor.git

# Changer le répertoire vers le projet
cd delivery-time-predictor

# [Optionnel mais recommandé] Créer et activer un environnement virtuel
# python -m venv venv
# source venv/bin/activate        # Sur Linux/macOS
# .\venv\Scripts\activate          # Sur Windows (PowerShell ou CMD)
```

### 2. Installation des Dépendances

Installez toutes les bibliothèques requises à partir du fichier `requirements.txt` :
```bash
pip install -r requirements.txt
```

### 3. Analyse et Exécution du Pipeline

**Visualiser l'EDA (Analyse Exploratoire des Données) :**
```bash
jupyter notebook notebook.ipynb
```

**Exécuter le Pipeline de Production :** Cette commande va charger les données, effectuer le nettoyage, entraîner les modèles, les optimiser et les évaluer selon les métriques MAE et R² score.
```bash
python pipeline.py
```

### 4. Lancer les Tests

Exécutez les tests unitaires pour vérifier la fiabilité du code et la performance du modèle :
```bash
pytest test_pipeline.py
# Ou simplement :
# pytest
```

---

**Auteur :** Ayoub Motei  
**Repository :** [delivery-time-predictor](https://github.com/AyoubMotei/delivery-time-predictor)