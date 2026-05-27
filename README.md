# 💄 Cosmo·Universe

> Chatbot RAG multilingue sur un catalogue cosmétiques — projet portfolio

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cosmo-universe-rag-2rjfzr8grmgy8zfylcntmz.streamlit.app/)

---

## Présentation

Cosmo·Universe est un chatbot de recommandation cosmétique capable de répondre en **français, coréen et anglais** — y compris quand l'utilisateur mélange les langues.

Ce projet illustre la mise en place d'un pipeline **RAG (Retrieval-Augmented Generation)** de bout en bout, de la construction du dataset jusqu'au déploiement en production.

---

## Fonctionnalités

- Recherche sémantique multilingue sur 91 produits cosmétiques
- Détection automatique de la langue (FR / KR / EN / code-switching)
- Filtres par prix, type de produit et marque
- Mémoire de conversation (5 derniers échanges)
- Connaissance complète du catalogue (marques, catégories, fourchette de prix)
- Sécurité : rate limiting, validation des inputs, protection contre les injections de prompt
- Interface K-beauty — style élégant et épuré

---

## Architecture

```
Question utilisateur (FR / KR / EN)
    → Détection de langue
    → Filtres metadata (prix, type, marque)
    → Recherche sémantique ChromaDB
    → Contexte injecté dans le prompt
    → LLM (GPT-4o-mini)
    → Réponse dans la langue de l'utilisateur
```

---

## Stack technique

| Composant | Outil |
|-----------|-------|
| Interface | Streamlit |
| Pipeline RAG | LangChain |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | text-embedding-3-small |
| Base vectorielle | ChromaDB |
| Déploiement | Streamlit Cloud |

---

## Lancer le projet en local

```bash
# 1. Cloner le repo
git clone https://github.com/ejlee94/cosmo-universe-rag.git
cd cosmo-universe-rag

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer la clé OpenAI
echo "OPENAI_API_KEY=sk-..." > .env

# 4. Lancer l'application
streamlit run app.py
```

> La base vectorielle ChromaDB est créée automatiquement au premier démarrage depuis `products.csv`.

---

## Dataset

91 produits cosmétiques fictifs répartis en 11 marques et 10 catégories.
Chaque produit contient une description en français, coréen et anglais.

---

## Améliorations futures

- Intégration d'un feedback utilisateur (Google Sheets)
- Évaluation automatique avec RAGAS
- Extension à d'autres langues (japonais, espagnol)
- Dataset réel via scraping de catalogues publics

---

## Auteure

Projet réalisé dans le cadre d'une démarche de portfolio Data / IA.

