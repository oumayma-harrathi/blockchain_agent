# 🚀 Blockchain Intelligence Agent

Un agent intelligent d'analyse et de recommandation de blockchains, utilisant **LangChain**, **Ollama**, **RAG local**, **NER personnalisé**, et une base de connaissances dynamique.

L'agent comprend les besoins techniques (TPS, frais, RGPD, hors-ligne, etc.) et recommande la blockchain optimale avec justification complète.

---

## 🧩 Architecture du Projet

Ce projet est conçu comme un **système modulaire** avec plusieurs composants interconnectés :

### 🔹 `blockchain_agent.py`
- **Cœur du système** : orchestre tout
- **Fonctionnalités :**
  - Analyse dynamique des besoins (NLP + regex)
  - Recherche web en temps réel (via DuckDuckGo)
  - RAG (Chroma + HuggingFace Embeddings) pour connaissances locales
  - Génération en streaming avec `phi3:mini` (Ollama)
  - Historique persistant
  - Cache intelligent pour éviter les appels redondants

### 🔹 `config.py` & `config.yaml`
- **Configuration centralisée :**
  - Chemins de fichiers
  - Paramètres Ollama
  - Modèle d'embedding
  - Timeout recherche web

### 🔹 `update_blockchains.py`
- Met à jour `blockchains.json` avec des données récentes (API ou données statiques)
- Base de connaissance pour le RAG

### 🔹 `train_ner.py` & `test_ner.py`
- Entraîne un modèle NER personnalisé pour détecter :
  - `BLOCKCHAIN`, `DOMAIN`, `METRIC`, `COMPLIANCE`, etc.
- Améliore la compréhension contextuelle

### 🔹 `app.py`
- Interface **Streamlit** pour une démo interactive
- Utilise `get_response_stream()` pour afficher la réponse en temps réel

---

## 📦 Prérequis & Installation

### 1. Installer Ollama

> Le modèle `phi3:mini` tourne en local via Ollama.

```bash
# Télécharger Ollama
# https://ollama.com/download

# Ou via CLI (Linux/Mac)
curl -fsSL https://ollama.com/install.sh | sh
```

**Télécharger le modèle `phi3:mini`**
```bash
ollama pull phi3:mini
```

✅ **Taille :** ~3.8 GB — léger et rapide.

### 2. Installation des Dépendances

```bash
# Créer un environnement virtuel (recommandé)
python -m venv venv
venv\Scripts\activate  # Windows
# ou source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le modèle spaCy
python -m spacy download en_core_web_sm
```

---

## ⚠️ Causes de Lenteurs & Correctifs Appliqués

| **Problème** | **Cause** | **Solution Appliquée** |
|--------------|-----------|------------------------|
| 🔻 Lenteur au démarrage | Chargement lent du modèle Ollama | Ajout de `keep_alive="5m"` pour garder le modèle en mémoire |
| 🔻 Temps de réponse long | Recherche web bloquante | Timeout à 8s + cache global (2h) |
| 🔻 RAG lent | Embedding séquentiel | Parallélisation via `ThreadPoolExecutor` |
| 🔻 Erreur `'type'` | Conflit LangChain | Verrouillage des versions dans `requirements.txt` |
| 🔻 Erreur `raw_`, `blockchain_` | Syntaxe incomplète | Correction des boucles `for bc in raw_data` → `blockchain_data` |

---

## 📄 Fichiers Clés

| **Fichier** | **Description** |
|-------------|-----------------|
| `blockchain_agent.py` | Agent principal (LLM + RAG + Web + NER) |
| `config.yaml` | Configuration centrale |
| `config.py` | Charge la config |
| `blockchains.json` | Base de données locale (entrée pour RAG) |
| `update_blockchains.py` | Met à jour la base |
| `train_ner.py` | Entraîne le modèle NER personnalisé |
| `test_ner.py` | Teste le NER |
| `app.py` | Interface Streamlit |
| `chroma_db/` | Base vectorielle (auto-générée) |
| `models/blockchain_ner_en/` | Modèle NER (auto-généré) |

---

## 🚀 Étapes pour Tester le Projet

### 1. Mettre à jour la base de données
```bash
python update_blockchains.py
```
**Génère :** `blockchains.json`

### 2. Entraîner le modèle NER
```bash
python train_ner.py
```
**Génère :** `models/blockchain_ner_en/`

### 3. Tester le NER
```bash
python test_ner.py
```
**Doit afficher :** des entités comme `Solana (BLOCKCHAIN)`, `NFT (DOMAIN)`, etc.

### 4. Lancer l'agent en CLI
```bash
python blockchain_agent.py
```
**Pose une question comme :**
> "Je veux créer une marketplace NFT avec des frais faibles et une haute performance."

### 5. Lancer l'interface Streamlit (optionnel)
```bash
streamlit run app.py
```
**Ouvre :** `http://localhost:8501`

---

## 🧪 Exemple de Question & Réponse

### **Entrée :**
> Une galerie d'art numérique cherche à : tokeniser des œuvres sous forme de NFTs dynamiques, offrir une protection des droits d'auteur via la blockchain, gérer les royalties automatiques à chaque revente, permettre l'achat multi-devise (ETH, USDC, carte), éviter les frais gas trop élevés. Quelle solution recommandez-vous ?

### **Sortie attendue :**
- ✅ Analyse des besoins (NFT, royalties, multi-devises, faible coût)
- ✅ Recherche web en temps réel
- ✅ RAG activé
- ✅ Recommandation argumentée (ex: **Polygon** ou **Solana**)
- ✅ Format structuré : objectif, analyse, justification, écosystème

---

## 📂 Structure du Projet

```
blockchain-agent/
├── blockchain_agent.py       # Agent principal
├── app.py                    # Interface Streamlit
├── config.yaml               # Configuration
├── config.py                 # Charge config
├── blockchains.json          # Base de données
├── conversation_history.json # Historique des conversations
├── update_blockchains.py     # Script de mise à jour
├── train_ner.py              # Entraînement NER
├── test_ner.py               # Test NER
├── requirements.txt          # Dépendances Python
├── chroma_db/                # Base vectorielle (auto-générée)
└── models/
    └── blockchain_ner_en/    # Modèle NER (auto-généré)
```

---

## ✅ Bon à Savoir

- 🚀 **Premier lancement :** Lent (embedding parallèle + téléchargement modèle)
- ⚡ **Lancements suivants :** Rapide grâce au **cache** (RAG, web, NER)
- 🌍 **Compatibilité :** Windows, Linux, Mac
- 🔒 **Confidentialité :** Aucune donnée envoyée à des serveurs externes (tout est local)
- 🧠 **Intelligence :** Combine LLM local, RAG, recherche web et NER personnalisé
- 📊 **Performance :** Optimisé pour des réponses rapides et précises

---

## 🔧 Configuration Avancée

Le fichier `config.yaml` permet de personnaliser :

```yaml
# Exemple de configuration
ollama:
  model: "phi3:mini"
  base_url: "http://localhost:11434"
  keep_alive: "5m"

embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  
web_search:
  timeout: 8
  cache_duration_hours: 2

paths:
  blockchain_db: "blockchains.json"
  conversation_history: "conversation_history.json"
  chroma_db: "chroma_db"
```

---

## 🐛 Dépannage

### Problème : Ollama ne répond pas
```bash
# Vérifier si Ollama est démarré
ollama list

# Redémarrer Ollama si nécessaire
ollama serve
```

### Problème : Erreur d'import LangChain
```bash
# Réinstaller avec les versions exactes
pip install -r requirements.txt --force-reinstall
```

### Problème : Modèle NER introuvable
```bash
# Re-entraîner le modèle
python train_ner.py
```

---

## 🐳 Dockerisation en Cours

Nous travaillons actuellement sur la **dockerisation complète** du projet pour le rendre **adaptable à n'importe quel système d'exploitation** (Windows, Linux, macOS).

La containerisation permettra :
- ✅ Installation en une seule commande
- ✅ Environnement isolé et reproductible  
- ✅ Déploiement simplifié sur serveurs
- ✅ Compatibilité garantie tous OS



