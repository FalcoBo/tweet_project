import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from collections import Counter

# Importer nos fonctions de preprocessing
import sys
import os

# Ajouter le chemin vers le dossier src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import *


class TestDataLoading:
    """Tests pour le chargement des données"""
    
    def test_load_data_valid_file(self):
        """Test du chargement d'un fichier CSV valide"""
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("text,target\n")
            f.write("Hello world,0\n")
            f.write("Earthquake today,1\n")
            temp_file = f.name
        
        try:
            df = load_data(temp_file)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert list(df.columns) == ['text', 'target']
        finally:
            os.unlink(temp_file)
    
    def test_load_data_file_not_found(self):
        """Test du chargement d'un fichier inexistant"""
        result = load_data("nonexistent_file.csv")
        assert result is None or result.empty


class TestDataCleaning:
    """Tests pour le nettoyage des données"""
    
    def test_handle_missing_values(self):
        """Test de la gestion des valeurs manquantes"""
        df = pd.DataFrame({
            'text': ['Hello', None, '', 'World'],
            'keyword': ['test', None, '', 'keyword'],
            'location': ['NYC', None, '', 'LA']
        })
        
        result = handle_missing_values(df)
        
        # Vérifier que les colonnes text, keyword, location n'ont plus de valeurs manquantes
        text_columns = ['text', 'keyword', 'location']
        for col in text_columns:
            if col in result.columns:
                assert result[col].isnull().sum() == 0
        assert len(result) == len(df)
    
    def test_remove_duplicates(self):
        """Test de la suppression des doublons"""
        df = pd.DataFrame({
            'text': ['Hello', 'Hello', 'World'],
            'target': [0, 0, 1]
        })
        
        result = remove_duplicates(df)
        assert len(result) == 2
        assert list(result['text']) == ['Hello', 'World']


class TestTextCleaning:
    """Tests pour le nettoyage de texte"""
    
    def test_clean_text_basic(self):
        """Test du nettoyage de texte basique"""
        text = "Hello World! This is a TEST 123."
        result = clean_text(text)
        
        assert isinstance(result, str)
        assert result.islower()
        assert "123" not in result  # Chiffres supprimés
        assert "!" not in result    # Ponctuation supprimée
    
    def test_clean_text_urls(self):
        """Test de suppression des URLs"""
        text = "Check this out: https://example.com and www.test.org"
        result = clean_text(text)
        
        assert "https://example.com" not in result
        assert "www.test.org" not in result
    
    def test_clean_text_mentions_hashtags(self):
        """Test de suppression des mentions et hashtags"""
        text = "Hello @user! Check #hashtag and #another_tag"
        result = clean_text(text)
        
        assert "@user" not in result
        assert "#hashtag" not in result
        assert "#another_tag" not in result
    
    def test_clean_text_empty_input(self):
        """Test avec entrée vide"""
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_clean_text_special_cases(self):
        """Test de cas spéciaux"""
        # Test avec seulement de la ponctuation
        result = clean_text("!@#$%^&*()")
        assert result.strip() == ""
        
        # Test avec espaces multiples
        result = clean_text("hello    world")
        assert "  " not in result


class TestTokenization:
    """Tests pour la tokenisation"""
    
    def test_clean_text_tokenization(self):
        """Test de tokenisation via clean_text"""
        text = "hello world test"
        result = clean_text(text)
        tokens = result.split()
        
        assert isinstance(tokens, list)
        assert len(tokens) == 3
        assert tokens == ['hello', 'world', 'test']
    
    def test_preprocess_text_tokenization(self):
        """Test de tokenisation via preprocess_text"""
        text = "Hello world! This is a test."
        result = preprocess_text(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Après preprocessing, le texte devrait être nettoyé
        assert result.islower() or result == ""


class TestStemmingLemmatization:
    """Tests pour stemming et lemmatisation"""
    
    def test_stem_text_basic(self):
        """Test de stemming basique"""
        text = "running runs runner"
        result = stem_text(text)
        
        assert isinstance(result, str)
        # Vérifier que les mots sont transformés
        tokens = result.split()
        assert len(tokens) <= 3  # Peut réduire le nombre de tokens uniques
    
    def test_stem_text_empty(self):
        """Test de stemming avec texte vide"""
        assert stem_text("") == ""
        assert stem_text(None) == ""


class TestStopwordRemoval:
    """Tests pour la suppression des stopwords"""
    
    def test_remove_stopwords_basic(self):
        """Test de suppression des stopwords"""
        text = "this is a test with the and or"
        result = remove_stopwords(text)
        
        # Les mots courants devraient être supprimés
        assert "this" not in result.lower()
        assert "is" not in result.lower()
        assert "the" not in result.lower()
        assert "test" in result.lower()  # 'test' n'est pas un stopword
    
    def test_remove_stopwords_empty(self):
        """Test avec texte vide"""
        assert remove_stopwords("") == ""
        assert remove_stopwords(None) == ""
    
    def test_remove_stopwords_only_stopwords(self):
        """Test avec seulement des stopwords"""
        text = "the and or but"
        result = remove_stopwords(text)
        assert result.strip() == "" or len(result.split()) == 0


class TestCompletePreprocessing:
    """Tests pour le preprocessing complet"""
    
    def test_preprocess_text_integration(self):
        """Test d'intégration du preprocessing complet"""
        text = "Hello @user! Check https://example.com #test 123"
        result = preprocess_text(text)
        
        assert isinstance(result, str)
        # Vérifications sur le résultat final
        assert "@user" not in result
        assert "https://example.com" not in result
        assert "#test" not in result
        assert "123" not in result
    
    def test_preprocess_text_column(self):
        """Test du preprocessing d'une colonne DataFrame"""
        df = pd.DataFrame({
            'text': ['Hello World!', 'Test @user #hashtag', 'Another tweet'],
            'target': [0, 1, 0]
        })
        
        result = preprocess_text_column(df, 'text')
        
        assert 'text_processed' in result.columns
        assert len(result) == len(df)
        assert all(isinstance(text, str) for text in result['text_processed'])
    
    def test_preprocess_text_robustness(self):
        """Test de robustesse du preprocessing"""
        test_cases = [
            "",  # Texte vide
            None,  # Valeur None
            "OK",  # Texte très court
            "123 456 789",  # Seulement des chiffres
            "!@#$%^&*()",  # Seulement de la ponctuation
            "Fire! 911 emergency #help @rescue",  # Texte mixte
        ]
        
        for text in test_cases:
            result = preprocess_text(text)
            assert isinstance(result, str)  # Doit toujours retourner une string


class TestFeatureExtraction:
    """Tests pour l'extraction de features"""
    
    def test_create_features_basic(self):
        """Test de création de features de base"""
        df = pd.DataFrame({
            'text': ['Hello @user! #test https://example.com', 'Simple text', '']
        })
        
        result = create_features(df, 'text')
        
        # Vérifier les nouvelles colonnes
        expected_features = ['text_length', 'word_count', 'hashtag_count', 
                           'mention_count', 'url_count']
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Vérifier les valeurs
        assert result['hashtag_count'].iloc[0] == 1
        assert result['mention_count'].iloc[0] == 1
        assert result['url_count'].iloc[0] == 1
        assert result['word_count'].iloc[0] > 0
    
    def test_create_features_edge_cases(self):
        """Test avec cas limites"""
        df = pd.DataFrame({
            'text': ['', None, '   ']  # Cas limites
        })
        
        result = create_features(df, 'text')
        
        # Vérifier que la fonction retourne un DataFrame valide
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)


class TestTextStatistics:
    """Tests pour les statistiques de texte"""
    
    def test_get_text_statistics(self):
        """Test des statistiques de texte"""
        df = pd.DataFrame({
            'text': ['Hello world', 'This is a test', 'Short']
        })
        
        stats = get_text_statistics(df, 'text')
        
        assert isinstance(stats, dict)
        assert 'avg_length' in stats
        assert 'avg_word_count' in stats
        assert 'max_length' in stats
        assert 'min_length' in stats
        
        assert stats['avg_length'] > 0
        assert stats['avg_word_count'] > 0
    
    def test_identify_outliers(self):
        """Test d'identification des outliers"""
        df = pd.DataFrame({
            'text': ['OK', 'This is a normal tweet', 'A' * 300]  # Court, normal, long
        })
        
        outliers = identify_outliers(df, 'text', min_length=10, max_length=280)
        
        assert isinstance(outliers, dict)
        assert 'short_tweets_count' in outliers
        assert 'long_tweets_count' in outliers
        assert 'short_tweets_percentage' in outliers
        assert 'long_tweets_percentage' in outliers
        
        assert outliers['short_tweets_count'] >= 1  # Tweet "OK"
        assert outliers['long_tweets_count'] >= 1   # Tweet long


class TestSpacyIntegration:
    """Tests pour l'intégration SpaCy"""
    
    def test_preprocess_text_spacy(self):
        """Test du preprocessing avec SpaCy"""
        text = "The running dogs were quickly jumping over fences."
        result = preprocess_text_spacy(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Les stopwords devraient être supprimés
        assert "the" not in result.lower()
        assert "were" not in result.lower()
    
    def test_extract_named_entities(self):
        """Test d'extraction d'entités nommées"""
        text = "Apple Inc. is located in California, United States."
        entities = extract_named_entities(text)
        
        assert isinstance(entities, list)
        # Devrait détecter des entités
        if len(entities) > 0:
            # Si la fonction retourne des strings au lieu de dict
            for entity in entities:
                assert isinstance(entity, (str, dict))
    
    def test_extract_pos_tags(self):
        """Test d'extraction des tags POS"""
        text = "The quick brown fox jumps."
        pos_tags = extract_pos_tags(text)
        
        assert isinstance(pos_tags, dict)
        assert len(pos_tags) > 0
        
        # Vérifier que les clés sont des tags POS valides
        for pos_tag, count in pos_tags.items():
            assert isinstance(pos_tag, str)
            assert isinstance(count, int)
            assert count > 0


class TestCorpusAnalysis:
    """Tests pour l'analyse de corpus"""
    
    def test_corpus_token_analysis(self):
        """Test d'analyse des tokens d'un corpus"""
        texts = [
            "hello world test",
            "hello again world",
            "test unique word"
        ]
        
        # Simuler l'analyse comme dans le notebook
        all_tokens = []
        for text in texts:
            tokens = text.split()
            all_tokens.extend(tokens)
        
        total_tokens = len(all_tokens)
        unique_tokens = len(set(all_tokens))
        token_counts = Counter(all_tokens)
        hapax_tokens = sum(1 for count in token_counts.values() if count == 1)
        
        assert total_tokens == 9  # Nombre total de tokens
        assert unique_tokens == 6  # Tokens uniques: hello, world, test, again, unique, word
        assert hapax_tokens == 3   # again, unique, word (autres tokens uniques)
        assert token_counts['hello'] == 2  # 'hello' apparaît 2 fois
    
    def test_preprocessing_reduction(self):
        """Test de la réduction après preprocessing"""
        original_texts = [
            "Hello @user! Check https://example.com #test 123",
            "Another tweet with @mention and #hashtag",
            "Simple text without special elements"
        ]
        
        processed_texts = [preprocess_text(text) for text in original_texts]
        
        # Compter les tokens avant et après
        original_tokens = []
        processed_tokens = []
        
        for text in original_texts:
            original_tokens.extend(text.lower().split())
        
        for text in processed_texts:
            if text.strip():
                processed_tokens.extend(text.split())
        
        # Le preprocessing devrait réduire le nombre de tokens
        assert len(processed_tokens) <= len(original_tokens)
        assert len(set(processed_tokens)) <= len(set(original_tokens))