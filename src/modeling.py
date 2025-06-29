"""
Module de modélisation pour la classification de tweets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


def train_model(X, y, test_size=0.2, random_state=42):
    """
    Entraîner un modèle de classification de tweets.
    
    Args:
        X: Textes préprocessés
        y: Labels cibles
        test_size: Proportion de données de test
        random_state: Graine aléatoire
        
    Returns:
        Modèle entraîné (Pipeline)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Pipeline avec TF-IDF et Logistic Regression
    model = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=random_state))
    ])
    
    # Entraînement
    model.fit(X_train, y_train)
    
    return model


def predict_tweets(model, texts):
    """
    Prédire les classes pour une liste de tweets.
    
    Args:
        model: Modèle entraîné
        texts: Liste de textes à classifier
        
    Returns:
        Array des prédictions
    """
    predictions = model.predict(texts)
    return predictions


def evaluate_model(y_true, y_pred):
    """
    Évaluer les performances du modèle.
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions du modèle
        
    Returns:
        Dictionnaire des métriques
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }
    
    return metrics


def optimize_model(X, y, cv=3):
    """
    Optimiser les hyperparamètres du modèle.
    
    Args:
        X: Textes préprocessés
        y: Labels cibles
        cv: Nombre de folds pour la validation croisée
        
    Returns:
        Meilleur modèle optimisé
    """
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    param_grid = {
        'vectorizer__max_features': [3000, 5000, 8000],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'classifier__C': [0.1, 1, 10]
    }
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_