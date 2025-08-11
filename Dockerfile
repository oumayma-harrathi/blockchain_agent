FROM python:3.11-slim

WORKDIR /app

# Installer curl pour Ollama
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Installer Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copier les dépendances
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copier le projet
COPY . .

# Exposer Streamlit
EXPOSE 8501

# Démarrer Ollama + Streamlit
CMD ["sh", "-c", "ollama serve & sleep 10 && ollama run phi3:mini & sleep 5 && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]