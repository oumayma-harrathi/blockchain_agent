FROM python:3.11-slim

WORKDIR /app

# Copier requirements et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copier le reste du projet
COPY . .

# Exposer le port de Streamlit
EXPOSE 8501

# Commande par défaut
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]