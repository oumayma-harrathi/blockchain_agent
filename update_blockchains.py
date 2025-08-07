# update_blockchains.py
import json
import requests
from datetime import datetime
import logging
from config import Config

# Charger la config
config = Config.load()

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_FILE = config.BLOCKCHAIN_DATA_FILE

# Simulons des appels API réels (tu peux remplacer par des vrais endpoints)
def fetch_real_data():
    # Exemple : API CoinGecko, Blockchain explorers, etc.
    return [
        {
            "name": "Ethereum",
            "type": "Public",
            "consensus": "Proof of Stake",
            "TPS": "15-30",
            "tx_cost": "$1.50 avg",
            "use_cases": ["DeFi", "NFTs", "Smart Contracts"],
            "pros": ["Sécurité", "Écosystème vaste", "EVM compatible"],
            "cons": ["Frais variables", "Congestion"]
        },
        {
            "name": "Solana",
            "type": "Public",
            "consensus": "Proof of History",
            "TPS": "50,000+",
            "tx_cost": "< $0.001",
            "use_cases": ["Gaming", "NFTs", "Paiements rapides"],
            "pros": ["Très rapide", "Frais négligeables", "Finalité rapide"],
            "cons": ["Centralisation partielle", "Historique d'outages"]
        },
        {
            "name": "Polygon",
            "type": "Layer 2",
            "consensus": "Proof of Stake",
            "TPS": "7,000",
            "tx_cost": "< $0.01",
            "use_cases": ["dApps", "NFTs", "Enterprise"],
            "pros": ["Compatible Ethereum", "Faible coût", "Évolutif"],
            "cons": ["Sécurité dépendante d'Ethereum"]
        }
    ]

def update_blockchains():
    try:
        data = fetch_real_data()
        data.append({"last_updated": datetime.now().isoformat()})

        with open(config.BLOCKCHAIN_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ {config.BLOCKCHAIN_DATA_FILE} mis à jour avec {len(data)-1} blockchains")
        logger.info(f"🔧 Base RAG sera régénérée au prochain lancement de l'agent.")

    except Exception as e:
        logger.error(f"❌ Échec de la mise à jour : {e}")

if __name__ == "__main__":
    update_blockchains()