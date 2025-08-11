# Votre Dockerfile devrait ressembler à :
FROM python:3.11-slim

WORKDIR /app

# Copier requirements
COPY requirements.txt .

# Installer dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Exposer le port
EXPOSE 8080

# Commande de démarrage
CMD ["python", "app.py"]