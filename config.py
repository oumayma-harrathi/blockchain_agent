# config.py
import yaml
import os
from dataclasses import dataclass

CONFIG_FILE = "config.yaml"

@dataclass
class Config:
    BLOCKCHAIN_DATA_FILE: str
    CHROMA_DB_DIR: str
    MODEL_NAME: str
    LOG_LEVEL: str
    MAX_HISTORY_ENTRIES: int
    HISTORY_FILE: str
    SEARCH_TIMEOUT: int
    SEARCH_MAX_RESULTS: int
    NER_MODEL_PATH: str
    USE_CUSTOM_NER: bool
    EMBEDDING_MODEL: str
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    OLLAMA_KEEP_ALIVE: str
    OLLAMA_TEMPERATURE: float
    OLLAMA_TOP_P: float

    @staticmethod
    def load(config_file: str = CONFIG_FILE):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"❌ {config_file} introuvable. Lance setup_config().")
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Config(**data)

# Exemple d'utilisation :
# config = Config.load()