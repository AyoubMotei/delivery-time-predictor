import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np

# --- FONCTIONS DE BASE ---

def load_data(file_path):
    """Charge les données depuis un fichier CSV."""
    print("-> Chargement des données.")
    return pd.read_csv(file_path)

def clean_data(df): 
    print("-> Nettoyage des données (Imputation Médiane/Mode).")
    # Imputation numérique
    median_experience = df['Courier_Experience_yrs'].median()
    df['Courier_Experience_yrs'] = df['Courier_Experience_yrs'].fillna(median_experience)
    
    # Imputation catégorielle (Weather, Traffic, Time_of_Day)
    categorical_cols_to_impute = ['Weather', 'Traffic_Level', 'Time_of_Day']
    for column in categorical_cols_to_impute:
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)
        
    # L'affichage des manquants est plus clair avec sum().sum()
    print(f"   Vérification des manquants après nettoyage: {df.isnull().sum().sum()}")
    return df

def split_features_target(df):
    """Sépare les caractéristiques (X) et la cible (y)."""
    X = df.drop(columns=['Delivery_Time_min', 'Order_ID'])
    y = df['Delivery_Time_min']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"-> Données divisées: {X_train.shape[0]} pour l'entraînement, {X_test.shape[0]} pour le test.")
    return X_train, X_test, y_train, y_test

def create_preprocessor():
    print("-> Création du ColumnTransformer (StandardScaler & OHE).")
    
    numerical_features = ['Distance_km', 'Courier_Experience_yrs', 'Preparation_Time_min']
    categorical_features = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Exclut Order_ID et la cible  
    )
    return preprocessor

def create_pipeline(preprocessor, model_type, select_k_best):
    """Crée un pipeline de prétraitement et de modélisation."""
      # Sélection du régresseur
    if model_type == 'rfr':
        regressor = RandomForestRegressor(random_state=42)
    elif model_type == 'svr':
        regressor = SVR()
    else:
        raise ValueError("Modèle inconnu. Utilisez 'rfr' ou 'svr'.")
    
     # Définition des étapes du pipeline
    steps = [('preprocessor', preprocessor)]
    
    # Ajout de la sélection de caractéristiques
    if select_k_best is not None:
        steps.append(('feature_selection', SelectKBest(score_func=f_regression, k=select_k_best)))
        
     # Ajout du régresseur
    steps.append(('regressor', regressor))
    
    # Création du pipeline
    pipeline = Pipeline(steps=steps)
    return pipeline

def train_and_optimize_model(X_train, y_train, preprocessor, model_type, param_grid):
    """Entraîne et optimise le modèle avec GridSearchCV."""
    
    # Créer le pipeline
    pipeline = create_pipeline(preprocessor, model_type, select_k_best=True)
    
    # Configurer GridSearchCV
    print(f"\n-> Démarrage de GridSearchCV pour {model_type.upper()}...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    
    # Entraîner le modèle
    print(f"-> Entraînement et optimisation du modèle ({model_type.upper()}).")
    grid_search.fit(X_train, y_train)
    
    # Meilleur modèle
    best_mae = -grid_search.best_score_  # Convertir en MAE positif
    
    print(f"   -> Meilleur MAE CV: {best_mae:.2f} min avec les paramètres: {grid_search.best_params_}")
    
    return grid_search

def evaluate_model(grid_search, X_test, y_test):
    """Évalue le modèle sur le jeu de test."""
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mae, r2, y_pred, best_model

# --- FONCTION PRINCIPALE ---

def run_pipeline(file_path):
    """Exécute l'ensemble du pipeline de bout en bout."""
    
    # Charger et nettoyer les données
    df = load_data(file_path)
    df = clean_data(df)
    
    # Séparer les caractéristiques et la cible
    X_train, X_test, y_train, y_test = split_features_target(df)
    
    
    # Créer le préprocesseur
    preprocessor = create_preprocessor()
    
    # Définir les grilles d'hyperparamètres pour chaque modèle
    param_grid_rfr = {
        'regressor__n_estimators': [50, 100],
        'regressor__max_depth': [None, 10, 20],
        'feature_selection__k': [5, 7, 'all'] 
    }
    
    param_grid_svr = {
        'regressor__C': [0.1, 1, 10],
        'regressor__kernel': ['rbf', 'linear'],
        'feature_selection__k': [5, 7, 'all']  
    }
    
    # Entraîner et optimiser le modèle Random Forest
    grid_search_rfr = train_and_optimize_model(X_train, y_train, preprocessor, 'rfr', param_grid_rfr)
    
    # Entraîner et optimiser le modèle SVR
    grid_search_svr = train_and_optimize_model(X_train, y_train, preprocessor, 'svr', param_grid_svr)
    
    # Évaluer les modèles sur le jeu de test
    mae_rfr, r2_rfr, y_pred_rfr, best_model_rfr = evaluate_model(grid_search_rfr, X_test, y_test)
    mae_svr, r2_svr, y_pred_svr, best_model_svr = evaluate_model(grid_search_svr, X_test, y_test)
    
    # Afficher les résultats
    print("\n--- Résultats sur le jeu de test ---")
    print(f"Random Forest Regressor - MAE: {mae_rfr:.2f}, R²: {r2_rfr:.2f}")
    print(f"Support Vector Regressor - MAE: {mae_svr:.2f}, R²: {r2_svr:.2f}")
    
    results = {
        'RFR': {'MAE': mae_rfr, 'R2': r2_rfr, 'Model': best_model_rfr},
        'SVR': {'MAE': mae_svr, 'R2': r2_svr, 'Model': best_model_svr}
    }
    
    final_model_name = 'SVR' if results['SVR']['MAE'] < results['RFR']['MAE'] else 'RFR'
    final_model = results[final_model_name]['Model']
    
    print("\n--- RÉSULTATS DE LA COMPARAISON FINALE (Jeu de Test) ---")
    print(f"RFR - MAE: {mae_rfr:.2f}, R²: {r2_rfr:.4f}")
    print(f"SVR - MAE: {mae_svr:.2f}, R²: {r2_svr:.4f}")
    print(f"Le modèle adopté est : {final_model_name} (MAE: {results[final_model_name]['MAE']:.2f})")
    
    return results, final_model


  
if __name__ == "__main__":
    run_pipeline('dataset.csv')
    