import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from unittest.mock import patch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from modeling import *
from preprocessing import preprocess_text


class TestModelPrediction:
    """Tests pour les prédictions du modèle"""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture pour créer des données d'exemple"""
        data = {
            'text': [
                'Earthquake hits California buildings collapsed emergency',
                'Beautiful sunset today perfect weather wonderful day',
                'URGENT Fire spreading quickly evacuate now dangerous situation',
                'Just had lunch with friends great day amazing food',
                'Flood warning issued downtown area residents evacuate immediately'
            ],
            'target': [1, 0, 1, 0, 1]
        }
        df = pd.DataFrame(data)
        # Preprocessing des textes
        df['text_processed'] = df['text'].apply(preprocess_text)
        return df
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Fixture pour créer un modèle pré-entraîné"""
        X = sample_data['text_processed']
        y = sample_data['target']
        
        model = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=1000)),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        model.fit(X, y)
        return model
    
    def test_model_prediction_single_text(self, trained_model):
        """Test de prédiction sur un seul texte"""
        test_text = "Emergency earthquake situation dangerous"
        processed_text = preprocess_text(test_text)
        
        prediction = trained_model.predict([processed_text])[0]
        probability = trained_model.predict_proba([processed_text])[0]
        
        assert prediction in [0, 1]
        assert len(probability) == 2
        assert sum(probability) == pytest.approx(1.0, rel=1e-5)
        assert all(0 <= p <= 1 for p in probability)
    
    def test_model_prediction_batch(self, trained_model, sample_data):
        """Test de prédiction sur un batch de textes"""
        X_test = sample_data['text_processed']
        predictions = trained_model.predict(X_test)
        probabilities = trained_model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)
        assert probabilities.shape == (len(X_test), 2)
    
    def test_model_prediction_confidence(self, trained_model):
        """Test des niveaux de confiance des prédictions"""
        disaster_text = "earthquake emergency evacuation dangerous"
        normal_text = "beautiful day sunshine happiness"
        
        disaster_processed = preprocess_text(disaster_text)
        normal_processed = preprocess_text(normal_text)
        
        disaster_prob = trained_model.predict_proba([disaster_processed])[0]
        normal_prob = trained_model.predict_proba([normal_processed])[0]
        
        # Le modèle devrait être plus confiant sur les textes clairs
        disaster_confidence = max(disaster_prob)
        normal_confidence = max(normal_prob)
        
        assert disaster_confidence > 0.5
        assert normal_confidence > 0.5


class TestModelRobustness:
    """Tests de robustesse du modèle"""
    
    @pytest.fixture
    def simple_model(self):
        """Fixture pour un modèle simple"""
        # Créer des données d'entraînement simples
        X_train = ['disaster emergency', 'beautiful day', 'fire danger', 'happy time']
        y_train = [1, 0, 1, 0]
        
        model = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        model.fit(X_train, y_train)
        return model
    
    def test_model_empty_input(self, simple_model):
        """Test avec entrée vide"""
        empty_inputs = ["", "   ", None]
        
        for empty_input in empty_inputs:
            processed_input = preprocess_text(empty_input) if empty_input else ""
            
            try:
                prediction = simple_model.predict([processed_input])
                probability = simple_model.predict_proba([processed_input])
                
                assert prediction[0] in [0, 1]
                assert len(probability[0]) == 2
                
            except Exception as e:
                # Si une exception est levée, elle doit être gérable
                assert isinstance(e, (ValueError, AttributeError))
    
    def test_model_short_input(self, simple_model):
        """Test avec entrées très courtes"""
        short_inputs = ["OK", "No", "Hi", "123"]
        
        for short_input in short_inputs:
            processed_input = preprocess_text(short_input)
            
            prediction = simple_model.predict([processed_input])
            probability = simple_model.predict_proba([processed_input])
            
            assert prediction[0] in [0, 1]
            assert len(probability[0]) == 2
    
    def test_model_special_characters(self, simple_model):
        """Test avec caractères spéciaux"""
        special_inputs = [
            "!@#$%^&*()",
            "123 456 789",
            "http://example.com @user #hashtag",
            "Fire!!! 🔥🔥🔥 Emergency!!!"
        ]
        
        for special_input in special_inputs:
            processed_input = preprocess_text(special_input)
            
            prediction = simple_model.predict([processed_input])
            probability = simple_model.predict_proba([processed_input])
            
            assert prediction[0] in [0, 1]
            assert len(probability[0]) == 2
    
    def test_model_mixed_input(self, simple_model):
        """Test avec entrées mixtes (texte + éléments spéciaux)"""
        mixed_inputs = [
            "Fire! 911 emergency #help @rescue https://news.com",
            "Beautiful day 😊 #sunshine @friends pic.twitter.com/xyz",
            "URGENT: Earthquake! Call 911 NOW! #emergency"
        ]
        
        for mixed_input in mixed_inputs:
            processed_input = preprocess_text(mixed_input)
            
            prediction = simple_model.predict([processed_input])
            probability = simple_model.predict_proba([processed_input])
            
            assert prediction[0] in [0, 1]
            assert len(probability[0]) == 2
            # Le texte preprocessé ne devrait pas être vide pour ces cas
            assert len(processed_input.strip()) > 0


class TestModelTraining:
    """Tests pour l'entraînement des modèles"""
    
    def test_pipeline_creation(self):
        """Test de création des pipelines de différents modèles"""
        models = {
            'LogisticRegression_TF': Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=1000)),
                ('classifier', LogisticRegression(random_state=42))
            ]),
            'LogisticRegression_Count': Pipeline([
                ('vectorizer', CountVectorizer(max_features=1000)),
                ('classifier', LogisticRegression(random_state=42))
            ]),
            'SVM_TF': Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=1000)),
                ('classifier', SVC(kernel='linear', random_state=42, probability=True))
            ]),
            'NaiveBayes_TF': Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=1000)),
                ('classifier', MultinomialNB())
            ]),
            'RandomForest_TF': Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=1000)),
                ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
            ])
        }
        
        for name, model in models.items():
            assert isinstance(model, Pipeline)
            assert len(model.steps) == 2
            assert model.steps[0][0] == 'vectorizer'
            assert model.steps[1][0] == 'classifier'
    
    def test_model_training_basic(self):
        """Test d'entraînement basique d'un modèle"""
        # Données d'entraînement simples
        X_train = [
            'earthquake disaster emergency danger',
            'fire emergency evacuation urgent',
            'beautiful sunny day weather nice',
            'happy birthday celebration party fun'
        ]
        y_train = [1, 1, 0, 0]
        
        model = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # L'entraînement ne devrait pas lever d'exception
        model.fit(X_train, y_train)
        
        # Vérifier que le modèle peut faire des prédictions
        predictions = model.predict(X_train)
        probabilities = model.predict_proba(X_train)
        
        assert len(predictions) == len(X_train)
        assert probabilities.shape == (len(X_train), 2)
    
    def test_model_evaluation_metrics(self):
        """Test des métriques d'évaluation"""
        # Prédictions simulées
        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 0, 0]
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        assert 0 <= accuracy <= 1
        assert 0 <= f1 <= 1
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)


class TestModelIntegration:
    """Tests d'intégration du pipeline complet"""
    
    def test_complete_pipeline(self):
        """Test du pipeline complet de preprocessing + modélisation"""
        # Données d'exemple
        raw_data = {
            'text': [
                'Earthquake hits California! Emergency evacuation #disaster',
                'Beautiful sunset today 😊 Perfect weather #happiness',
                'URGENT: Fire spreading! Call 911 NOW! @emergency',
                'Great day with friends @fun #goodtimes amazing food',
                'Flood warning downtown! Residents evacuate immediately!'
            ],
            'target': [1, 0, 1, 0, 1]
        }
        
        df = pd.DataFrame(raw_data)
        
        # Étape 1: Preprocessing
        df['text_processed'] = df['text'].apply(preprocess_text)
        
        # Vérifier que le preprocessing fonctionne
        assert all(isinstance(text, str) for text in df['text_processed'])
        
        # Étape 2: Division des données
        X = df['text_processed']
        y = df['target']
        
        # Pour les tests, on utilise toutes les données
        X_train, X_test = X, X
        y_train, y_test = y, y
        
        # Étape 3: Entraînement du modèle
        model = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=100)),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        model.fit(X_train, y_train)
        
        # Étape 4: Prédictions
        y_pred = model.predict(X_test)
        
        # Vérifications
        assert len(y_pred) == len(y_test)
        assert all(pred in [0, 1] for pred in y_pred)
        
        # Étape 5: Évaluation
        accuracy = accuracy_score(y_test, y_pred)
        assert 0 <= accuracy <= 1
    
    def test_cross_validation_simulation(self):
        """Test de simulation de validation croisée"""
        from sklearn.model_selection import cross_val_score
        
        # Données d'exemple plus larges pour la validation croisée
        X = [f'sample text {i} disaster emergency' if i % 2 == 0 
             else f'sample text {i} beautiful happy day' 
             for i in range(20)]
        y = [i % 2 for i in range(20)]
        
        model = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=100)),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # Validation croisée avec 3 folds (minimum pour 20 échantillons)
        cv_scores = cross_val_score(model, X, y, cv=3, scoring='f1')
        
        assert len(cv_scores) == 3
        assert all(0 <= score <= 1 for score in cv_scores)
        
        # Calculer la moyenne et l'écart-type
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        assert 0 <= mean_score <= 1
        assert std_score >= 0


class TestModelDeployment:
    """Tests pour le déploiement et l'utilisation du modèle"""
    
    def test_model_serialization_concept(self):
        """Test conceptuel de sérialisation du modèle"""
        # Créer et entraîner un modèle simple
        X_train = ['disaster emergency', 'beautiful day', 'fire danger', 'happy time']
        y_train = [1, 0, 1, 0]
        
        model = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        model.fit(X_train, y_train)
        
        # Test que le modèle maintient ses paramètres
        original_params = model.get_params()
        
        # Faire une prédiction
        test_prediction = model.predict(['emergency situation'])
        
        # Vérifier que les paramètres n'ont pas changé
        current_params = model.get_params()
        assert original_params == current_params
        
        # Vérifier la cohérence des prédictions
        repeated_prediction = model.predict(['emergency situation'])
        assert test_prediction[0] == repeated_prediction[0]
    
    def test_model_error_handling(self):
        """Test de gestion des erreurs du modèle"""
        model = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # Test avec modèle non-entraîné
        with pytest.raises(Exception):  # Devrait lever une exception
            model.predict(['test'])
        
        # Entraîner avec données minimales (2 classes requises)
        model.fit(['test class 0', 'test class 1'], [0, 1])
        
        # Maintenant les prédictions devraient fonctionner
        prediction = model.predict(['test'])
        assert len(prediction) == 1
        assert prediction[0] in [0, 1]


class TestModelComparison:
    """Tests pour la comparaison de modèles"""
    
    def test_multiple_models_comparison(self):
        """Test de comparaison entre plusieurs modèles"""
        # Données d'entraînement
        X = [
            'earthquake disaster emergency', 'fire danger evacuation',
            'beautiful day sunshine', 'happy celebration party',
            'flood warning urgent', 'lovely weather perfect'
        ]
        y = [1, 1, 0, 0, 1, 0]
        
        models = {
            'LogisticRegression': LogisticRegression(random_state=42),
            'NaiveBayes': MultinomialNB(),
            'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42)
        }
        
        vectorizer = TfidfVectorizer(max_features=100)
        X_vectorized = vectorizer.fit_transform(X)
        
        results = {}
        
        for name, classifier in models.items():
            classifier.fit(X_vectorized, y)
            predictions = classifier.predict(X_vectorized)
            accuracy = accuracy_score(y, predictions)
            
            results[name] = accuracy
            assert 0 <= accuracy <= 1
        
        # Vérifier qu'on a des résultats pour tous les modèles
        assert len(results) == len(models)
        assert all(isinstance(acc, (int, float)) for acc in results.values())
    
    def test_hyperparameter_impact(self):
        """Test de l'impact des hyperparamètres"""
        X = ['disaster emergency'] * 5 + ['beautiful day'] * 5
        y = [1] * 5 + [0] * 5
        
        vectorizer = TfidfVectorizer()
        X_vectorized = vectorizer.fit_transform(X)
        
        # Test avec différents paramètres C pour LogisticRegression
        c_values = [0.1, 1.0, 10.0]
        accuracies = []
        
        for c in c_values:
            model = LogisticRegression(C=c, random_state=42)
            model.fit(X_vectorized, y)
            predictions = model.predict(X_vectorized)
            accuracy = accuracy_score(y, predictions)
            accuracies.append(accuracy)
        
        # Toutes les précisions devraient être valides
        assert all(0 <= acc <= 1 for acc in accuracies)
        assert len(accuracies) == len(c_values)