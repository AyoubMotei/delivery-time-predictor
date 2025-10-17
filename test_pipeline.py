import pandas as pd
import numpy as np
import pytest
from pipeline import load_data, clean_data, split_features_target, create_preprocessor,run_pipeline


MAE_SEUIL_MAX = 8.0 # MAE maximale tolérée pour le meilleur modèle

def test_format_dimention():
    """
    Vérifie les dimensions des données et la transformation du préprocesseur.
    """

    df=load_data('dataset.csv')
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_features_target(df)
    preprocessor = create_preprocessor()
    
    #  Vérifier les dimensions de base
    assert df.shape[0] > 0, "Le DataFrame nettoyé est vide."
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0, "Les ensembles d'entraînement/test sont vides."

    #  Vérifier la transformation (qui inclut OHE et StandardScaler)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Vérifier le nombre de lignes (doit rester identique)
    assert X_train_transformed.shape[0] == X_train.shape[0]
    
    # Vérifier l'augmentation du nombre de colonnes (due à OneHotEncoder)
    assert X_train_transformed.shape[1] > X_train.shape[1], "Le nombre de caractéristiques n'a pas augmenté après OHE." 
    
    # Assurer l'absence de NaN après prétraitement
    # C'est crucial car les modèles SVR/RFR ne gèrent pas les NaN
    assert np.isnan(X_train_transformed).sum() == 0, "Des NaN ont été introduits ou n'ont pas été gérés par le préprocesseur."



def test_seuil_mae():
    """
    Teste que la MAE du modèle sélectionné est inférieure au seuil défini par le métier.
    """
    
    # Exécuter le pipeline complet
    results, final_model = run_pipeline('dataset.csv')
    
    # Identifier le meilleur modèle basé sur les résultats (MAE sur le jeu de test)
    mae_rfr = results['RFR']['MAE']
    mae_svr = results['SVR']['MAE']
    
    # Déterminer la MAE du modèle final adopté
    final_mae = min(mae_rfr, mae_svr)
    
    #  Vérifier que la MAE du meilleur modèle est inférieure au seuil maximal
    assert final_mae < MAE_SEUIL_MAX, (
        f"La MAE finale ({final_mae:.2f} min) est supérieure au seuil maximal de {MAE_SEUIL_MAX:.2f} min."
    )












