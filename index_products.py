"""
Étape 2 — Indexation des produits cosmétiques dans ChromaDB
Stack : OpenAI text-embedding-3-small + ChromaDB
1 produit = 1 document (FR + EN + KR concaténés)
"""

import os
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CSV_PATH       = "products.csv"
CHROMA_PATH    = "./chroma_db"
COLLECTION_NAME = "cosmetiques_multilingue"

# ─────────────────────────────────────────────
# 2. Chargement du dataset
# ─────────────────────────────────────────────

df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
print(f"⚠️  Lignes ignorées — {100 - len(df)} produits sur 100 chargés")

# ─────────────────────────────────────────────
# 3. Construction des documents à indexer
#    Stratégie : concaténer FR + EN + KR
#    → 1 vecteur par produit qui capture les 3 langues
# ─────────────────────────────────────────────

def build_document(row: pd.Series) -> str:
    """
    Construit le texte à vectoriser pour un produit.
    On inclut toutes les langues + métadonnées clés
    pour que la recherche fonctionne quelle que soit la langue.
    """
    return (
        f"Produit: {row['nom']} | Product: {row['nom']} | 제품: {row['nom']}\n"
        f"Marque: {row['marque']} | Brand: {row['marque']} | 브랜드: {row['marque']}\n"
        f"Catégorie: {row['categorie']} | Category: {row['categorie']}\n"
        f"Prix: {row['prix_eur']}€\n"
        f"Caractéristiques: {row['caracteristiques']}\n"
        f"FR: {row['description_fr']}\n"
        f"EN: {row['description_en']}\n"
        f"KR: {row['description_kr']}"
    )

documents = [build_document(row) for _, row in df.iterrows()]

# ─────────────────────────────────────────────
# 4. Métadonnées — filtrables dans ChromaDB
# ─────────────────────────────────────────────

# ✅ Nouveau code — avec type_produit
metadatas = [
    {
        "nom":            row["nom"],
        "marque":         row["marque"],
        "categorie":      row["categorie"],
        "prix_eur":       float(row["prix_eur"]),
        "caracteristiques": row["caracteristiques"],
        "description_fr": row["description_fr"],
        "description_en": row["description_en"],
        "description_kr": row["description_kr"],
        "type_produit":   row["type_produit"],
    }
    for _, row in df.iterrows()
]

# IDs uniques pour chaque produit
ids = [f"product_{i}" for i in range(len(df))]

# ─────────────────────────────────────────────
# 5. Initialisation ChromaDB + modèle embedding
# ─────────────────────────────────────────────

# Modèle multilingue OpenAI — gère FR, EN, KR nativement
embedding_function = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

# Client persistant — les données survivent au redémarrage
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Supprime la collection si elle existe déjà (utile pour re-indexer)
try:
    client.delete_collection(COLLECTION_NAME)
    print("🗑️  Ancienne collection supprimée")
except:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}  # similarité cosinus — standard pour le texte
)

print(f"✅ Collection '{COLLECTION_NAME}' créée")

# ─────────────────────────────────────────────
# 6. Indexation par batch
#    → évite les timeouts sur l'API OpenAI
# ─────────────────────────────────────────────

BATCH_SIZE = 20

for i in range(0, len(documents), BATCH_SIZE):
    batch_docs  = documents[i:i+BATCH_SIZE]
    batch_meta  = metadatas[i:i+BATCH_SIZE]
    batch_ids   = ids[i:i+BATCH_SIZE]

    collection.add(
        documents=batch_docs,
        metadatas=batch_meta,
        ids=batch_ids
    )
    print(f"  📦 Batch {i//BATCH_SIZE + 1} indexé ({len(batch_docs)} produits)")

print(f"\n✅ Indexation terminée — {collection.count()} produits dans ChromaDB")

# ─────────────────────────────────────────────
# 7. Test rapide — vérification
# ─────────────────────────────────────────────

print("\n🔍 Test de recherche rapide...")

# Test en français
results_fr = collection.query(
    query_texts=["crème hydratante pour peau sèche"],
    n_results=2
)
print("\n🇫🇷 Résultats FR — 'crème hydratante pour peau sèche':")
for meta in results_fr["metadatas"][0]:
    print(f"  → {meta['nom']} ({meta['marque']}) — {meta['prix_eur']}€")

# Test en anglais
results_en = collection.query(
    query_texts=["anti-aging serum for wrinkles"],
    n_results=2
)
print("\n🇬🇧 Résultats EN — 'anti-aging serum for wrinkles':")
for meta in results_en["metadatas"][0]:
    print(f"  → {meta['nom']} ({meta['marque']}) — {meta['prix_eur']}€")

# Test en coréen
results_kr = collection.query(
    query_texts=["건조한 피부를 위한 보습 크림"],
    n_results=2
)
print("\n🇰🇷 Résultats KR — '건조한 피부를 위한 보습 크림':")
for meta in results_kr["metadatas"][0]:
    print(f"  → {meta['nom']} ({meta['marque']}) — {meta['prix_eur']}€")
