# train_ner.py
import spacy
import random
from pathlib import Path
from config import Config

config = Config.load()

TRAIN_DATA = [
    {"text": "Je veux une blockchain comme Solana pour un jeu NFT avec 10k TPS.", "entities": [(25, 32, "BLOCKCHAIN"), (47, 52, "DOMAIN"), (68, 72, "METRIC")]},
    {"text": "Ethereum est trop lent pour les paiements rapides.", "entities": [(0, 8, "BLOCKCHAIN"), (38, 48, "DOMAIN")]},
    {"text": "Nous avons besoin d'une solution blockchain pour la santé avec RGPD.", "entities": [(45, 52, "DOMAIN"), (65, 70, "COMPLIANCE")]},
    {"text": "Cardano est une blockchain publique avec consensus Ouroboros.", "entities": [(0, 7, "BLOCKCHAIN"), (24, 31, "TYPE"), (59, 70, "CONSENSUS")]},
    {"text": "Avalanche offre 4500 TPS avec une latence de 2 secondes.", "entities": [(0, 8, "BLOCKCHAIN"), (21, 29, "METRIC"), (47, 61, "LATENCY")]},
    {"text": "Polygon est parfait pour les dApps avec faible coût et multi-devise.", "entities": [(0, 7, "BLOCKCHAIN"), (36, 48, "COST"), (53, 66, "PAYMENT")]},
]

def train_ner_model(output_dir=None, n_iter=100):
    output_dir = output_dir or config.NER_MODEL_PATH
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Ajouter les labels
    labels = set(ent[2] for example in TRAIN_DATA for ent in example["entities"])
    for label in labels:
        ner.add_label(label)

    # Entraînement
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for example in TRAIN_DATA:
            doc = nlp.make_doc(example["text"])
            annot = {"entities": example["entities"]}
            example = spacy.training.Example.from_dict(doc, annot)
            nlp.update([example], sgd=optimizer, losses=losses, drop=0.5)
        if itn % 20 == 0:
            print(f"Iter {itn}, Losses: {losses}")

    # Sauvegarder
    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_path)
    print(f"✅ Modèle NER sauvegardé dans {output_dir}")

if __name__ == "__main__":
    train_ner_model()