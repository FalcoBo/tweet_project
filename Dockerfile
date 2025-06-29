# Utiliser Python 3.12 comme image de base
FROM python:3.12-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Définir les variables d'environnement
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements.txt
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger le modèle SpaCy anglais
RUN python -m spacy download en_core_web_sm

# Copier le code source du projet
COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/
COPY notebooks/ ./notebooks/
COPY pytest.ini .
COPY main.py .

# Créer un utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Commande par défaut : exécuter les tests
CMD ["python", "-m", "pytest", "tests/", "-v"]