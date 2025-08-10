# test_ner.py
import spacy
from config import Config

config = Config.load()

try:
    nlp = spacy.load(config.NER_MODEL_PATH)
    print(f"✅ Modèle chargé depuis {config.NER_MODEL_PATH}")
except OSError:
    print(f"❌ Impossible de charger le modèle. Lance d'abord train_ner.py")
    exit(1)

text = "Solana is great for NFT gaming with 50k TPS and low latency. Ethereum has high gas fees."
doc = nlp(text)

print(f"\nText: {text}")
print("-" * 50)
for ent in doc.ents:
    print(f"→ {ent.text:<20} ({ent.label_:<10}) [pos: {ent.start_char}-{ent.end_char}]")
print("-" * 50)