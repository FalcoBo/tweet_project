import pandas as pd
import numpy as np
import re 
from typing import List, Dict
import nltk
import string


nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer



def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Preprocess the data from the given file path.
    
    Args:
        file_path (str): Path to the CSV file containing the data.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(file_path)
    
    # Appliquer toutes les étapes de preprocessing
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = preprocess_text_column(df)
    df = create_features(df)
    
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gérer les valeurs manquantes dans le DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame d'entrée
        
    Returns:
        pd.DataFrame: DataFrame avec les valeurs manquantes traitées
    """
    df_copy = df.copy()
    
    # Remplacer les valeurs manquantes dans les colonnes textuelles
    text_columns = ['text', 'keyword', 'location']
    for col in text_columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna('')
    
    return df_copy

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprimer les doublons du DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame d'entrée
        
    Returns:
        pd.DataFrame: DataFrame sans doublons
    """
    initial_count = len(df)
    df_clean = df.drop_duplicates()
    final_count = len(df_clean)
    
    print(f"Doublons supprimés: {initial_count - final_count}")
    return df_clean

def clean_text(text: str) -> str:
    """
    Nettoyer le texte en supprimant les éléments indésirables.
    
    Args:
        text (str): Texte à nettoyer
        
    Returns:
        str: Texte nettoyé
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer les URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Supprimer les mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Supprimer les hashtags (#hashtag)
    text = re.sub(r'#\w+', '', text)
    
    # Supprimer les chiffres
    text = re.sub(r'\d+', '', text)
    
    # Supprimer la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text: str, language: str = 'english') -> str:
    """
    Supprimer les mots vides du texte.
    
    Args:
        text (str): Texte d'entrée
        language (str): Langue pour les mots vides
        
    Returns:
        str: Texte sans mots vides
    """
    if pd.isna(text) or text == '':
        return ''
    
    stop_words = set(stopwords.words(language))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)

def stem_text(text: str) -> str:
    """
    Appliquer le stemming au texte.
    
    Args:
        text (str): Texte d'entrée
        
    Returns:
        str: Texte avec stemming appliqué
    """
    if pd.isna(text) or text == '':
        return ''
    
    stemmer = PorterStemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    
    return ' '.join(stemmed_words)

def preprocess_text(text: str, remove_stops: bool = True, do_stem: bool = True) -> str:
    """
    Fonction principale de preprocessing du texte.
    
    Args:
        text (str): Texte à traiter
        remove_stops (bool): Supprimer les mots vides
        do_stem (bool): Appliquer le stemming
        
    Returns:
        str: Texte préprocessé
    """
    if pd.isna(text):
        return ''
    
    # Nettoyer le texte
    text = clean_text(text)
    
    # Supprimer les mots vides si demandé
    if remove_stops:
        text = remove_stopwords(text)
    
    # Appliquer le stemming si demandé
    if do_stem:
        text = stem_text(text)
    
    return text

def preprocess_text_column(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Appliquer le preprocessing à une colonne de texte du DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame d'entrée
        text_column (str): Nom de la colonne contenant le texte
        
    Returns:
        pd.DataFrame: DataFrame avec colonne de texte préprocessée
    """
    if text_column not in df.columns:
        print(f"Colonne '{text_column}' non trouvée dans le DataFrame")
        return df
    
    df_copy = df.copy()
    df_copy[f'{text_column}_processed'] = df_copy[text_column].apply(preprocess_text)
    
    return df_copy

def create_features(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Créer des features supplémentaires à partir du texte.
    
    Args:
        df (pd.DataFrame): DataFrame d'entrée
        text_column (str): Nom de la colonne contenant le texte
        
    Returns:
        pd.DataFrame: DataFrame avec features supplémentaires
    """
    if text_column not in df.columns:
        print(f"Colonne '{text_column}' non trouvée dans le DataFrame")
        return df
    
    df_copy = df.copy()
    
    # Longueur du texte en caractères
    df_copy['text_length'] = df_copy[text_column].str.len()
    
    # Nombre de mots
    df_copy['word_count'] = df_copy[text_column].str.split().str.len()
    
    # Nombre de caractères uniques
    df_copy['unique_chars'] = df_copy[text_column].apply(lambda x: len(set(str(x))) if pd.notna(x) else 0)
    
    # Nombre de hashtags
    df_copy['hashtag_count'] = df_copy[text_column].apply(
        lambda x: len(re.findall(r'#\w+', str(x))) if pd.notna(x) else 0
    )
    
    # Nombre de mentions
    df_copy['mention_count'] = df_copy[text_column].apply(
        lambda x: len(re.findall(r'@\w+', str(x))) if pd.notna(x) else 0
    )
    
    # Nombre d'URLs
    df_copy['url_count'] = df_copy[text_column].apply(
        lambda x: len(re.findall(r'http\S+|www\S+|https\S+', str(x))) if pd.notna(x) else 0
    )
    
    # Pourcentage de caractères en majuscules
    df_copy['upper_case_ratio'] = df_copy[text_column].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if pd.notna(x) and len(str(x)) > 0 else 0
    )
    
    # Nombre de points d'exclamation
    df_copy['exclamation_count'] = df_copy[text_column].apply(
        lambda x: str(x).count('!') if pd.notna(x) else 0
    )
    
    # Nombre de points d'interrogation
    df_copy['question_count'] = df_copy[text_column].apply(
        lambda x: str(x).count('?') if pd.notna(x) else 0
    )
    
    return df_copy

def load_data(file_path: str) -> pd.DataFrame:
    """
    Charger les données depuis un fichier CSV.
    
    Args:
        file_path (str): Chemin vers le fichier CSV
        
    Returns:
        pd.DataFrame: DataFrame chargé
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Données chargées avec succès: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df
    except FileNotFoundError:
        print(f"Fichier non trouvé: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return pd.DataFrame()

def get_text_statistics(df: pd.DataFrame, text_column: str = 'text') -> dict:
    """
    Obtenir des statistiques sur la colonne de texte.
    
    Args:
        df (pd.DataFrame): DataFrame d'entrée
        text_column (str): Nom de la colonne contenant le texte
        
    Returns:
        dict: Dictionnaire avec les statistiques
    """
    if text_column not in df.columns:
        return {}
    
    text_lengths = df[text_column].str.len()
    word_counts = df[text_column].str.split().str.len()
    
    stats = {
        'total_tweets': len(df),
        'avg_length': text_lengths.mean(),
        'median_length': text_lengths.median(),
        'min_length': text_lengths.min(),
        'max_length': text_lengths.max(),
        'avg_word_count': word_counts.mean(),
        'median_word_count': word_counts.median(),
        'min_word_count': word_counts.min(),
        'max_word_count': word_counts.max()
    }
    
    return stats

def identify_outliers(df: pd.DataFrame, text_column: str = 'text', 
                     min_length: int = 10, max_length: int = 280) -> dict:
    """
    Identifier les tweets outliers (trop courts ou trop longs).
    
    Args:
        df (pd.DataFrame): DataFrame d'entrée
        text_column (str): Nom de la colonne contenant le texte
        min_length (int): Longueur minimale
        max_length (int): Longueur maximale
        
    Returns:
        dict: Informations sur les outliers
    """
    if text_column not in df.columns:
        return {}
    
    text_lengths = df[text_column].str.len()
    short_tweets = df[text_lengths < min_length]
    long_tweets = df[text_lengths > max_length]
    
    outliers_info = {
        'short_tweets_count': len(short_tweets),
        'long_tweets_count': len(long_tweets),
        'short_tweets_percentage': (len(short_tweets) / len(df)) * 100,
        'long_tweets_percentage': (len(long_tweets) / len(df)) * 100,
        'short_tweets_examples': short_tweets[text_column].head(3).tolist(),
        'long_tweets_examples': long_tweets[text_column].head(3).tolist()
    }
    
    return outliers_info