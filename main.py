#!/usr/bin/env python3
"""
Script principal pour le projet de classification de tweets.
Exécute le preprocessing et les tests.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Ajouter le dossier src au path Python
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def run_preprocessing():
    """Exécute le pipeline de preprocessing sur les données de test."""
    try:
        from preprocessing import load_data, preprocess_text_column, create_features
        
        # Chemin vers les données
        data_path = os.path.join(current_dir, 'data', 'tweets.csv')
        
        if not os.path.exists(data_path):
            print(f"Fichier de données non trouvé: {data_path}")
            return False
        
        print("Chargement des données...")
        df = load_data(data_path)
        print(f"Données chargées: {len(df)} lignes")
        
        print("Preprocessing du texte...")
        df_processed = preprocess_text_column(df, 'text')
        print("Preprocessing terminé")
        
        print("Extraction des features...")
        df_with_features = create_features(df_processed, 'text_processed')
        print("Features extraites")
        
        # Afficher quelques statistiques
        print(f"\nStatistiques:")
        print(f"- Nombre total de tweets: {len(df_with_features)}")
        if 'target' in df_with_features.columns:
            print(f"- Distribution des classes: {df_with_features['target'].value_counts().to_dict()}")
        print(f"- Colonnes générées: {list(df_with_features.columns)}")
        
        return True
        
    except Exception as e:
        print(f"Erreur lors du preprocessing: {e}")
        return False


def run_tests():
    """Exécute les tests unitaires."""
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'tests/', '-v'
        ], cwd=current_dir, capture_output=True, text=True)
        
        print("Résultats des tests:")
        print(result.stdout)
        if result.stderr:
            print("Erreurs:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Erreur lors de l'exécution des tests: {e}")
        return False


def main():
    """Fonction principale."""
    print("=== Pipeline de Classification de Tweets ===\n")
    
    success = True
    
    # Exécuter le preprocessing
    print("1. Exécution du preprocessing...")
    if not run_preprocessing():
        success = False
    
    print("\n" + "="*50 + "\n")
    
    # Exécuter les tests
    print("2. Exécution des tests...")
    if not run_tests():
        success = False
    
    print("\n" + "="*50 + "\n")
    
    if success:
        print("Pipeline exécuté avec succès.")
    else:
        print("Des erreurs ont été détectées.")
        sys.exit(1)


if __name__ == "__main__":
    main()