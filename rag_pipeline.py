"""
Étape 3 — Pipeline RAG multilingue FR / EN / KR
Stack : LangChain + OpenAI + ChromaDB
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY")
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "cosmetiques_multilingue"
TOP_K           = 5  # nombre de produits récupérés

# ─────────────────────────────────────────────
# 2. Connexion à ChromaDB
# ─────────────────────────────────────────────

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

print(f"✅ ChromaDB connecté — {vectorstore._collection.count()} produits")

# ─────────────────────────────────────────────
# 3. Initialisation du LLM
# ─────────────────────────────────────────────

llm = ChatOpenAI(
    model="gpt-4o-mini",       # rapide et économique
    temperature=0.3,           # réponses précises, peu aléatoires
    openai_api_key=OPENAI_API_KEY
)

# ─────────────────────────────────────────────
# 4. Prompt template multilingue
#    Le LLM détecte la langue et répond en conséquence
#    y compris si l'utilisateur mélange les langues (code-switching)
# ─────────────────────────────────────────────

PROMPT_TEMPLATE = """
Tu es un assistant expert en cosmétiques pour une boutique haut de gamme franco-coréenne.

INSTRUCTIONS :
- Détecte la langue de la question de l'utilisateur
- Réponds OBLIGATOIREMENT dans la même langue que la question :
    * Question en français → réponse en français UNIQUEMENT
    * Question en anglais → réponse en anglais UNIQUEMENT
    * Question en coréen → réponse en coréen UNIQUEMENT
    * Question mélangée FR+KR → réponds dans les deux langues naturellement
    * Question mélangée FR+EN → réponds dans les deux langues naturellement
    * Question mélangée EN+KR → réponds dans les deux langues naturellement
- Base ta réponse UNIQUEMENT sur les produits fournis dans le contexte
- Si aucun produit ne correspond, dis-le honnêtement
- Mentionne toujours le nom du produit, la marque et le prix dans ta réponse
- Sois chaleureux et professionnel, comme un conseiller en boutique de luxe

PRODUITS DISPONIBLES :
{context}

QUESTION : {question}

RÉPONSE :
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# ─────────────────────────────────────────────
# 5. Fonction de recherche dans ChromaDB
# ─────────────────────────────────────────────

def retrieve_products(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Recherche les produits les plus pertinents dans ChromaDB.
    Fonctionne quelle que soit la langue de la requête.
    """
    results = vectorstore.similarity_search(query, k=top_k)
    return [doc.metadata for doc in results]

# ─────────────────────────────────────────────
# 6. Formatage du contexte pour le LLM
# ─────────────────────────────────────────────

def format_context(products: list[dict], query: str) -> str:
    """
    Formate les produits récupérés en texte structuré.
    Inclut la description dans les 3 langues pour que le LLM
    puisse répondre dans la bonne langue.
    """
    context_parts = []

    for i, p in enumerate(products, 1):
        product_text = (
            f"Produit {i}:\n"
            f"  Nom: {p.get('nom', 'N/A')}\n"
            f"  Marque: {p.get('marque', 'N/A')}\n"
            f"  Catégorie: {p.get('categorie', 'N/A')}\n"
            f"  Prix: {p.get('prix_eur', 'N/A')}€\n"
            f"  Caractéristiques: {p.get('caracteristiques', 'N/A')}\n"
            f"  Description FR: {p.get('description_fr', 'N/A')}\n"
            f"  Description EN: {p.get('description_en', 'N/A')}\n"
            f"  Description KR: {p.get('description_kr', 'N/A')}\n"
        )
        context_parts.append(product_text)

    return "\n".join(context_parts)

# ─────────────────────────────────────────────
# 7. Fonction principale RAG
# ─────────────────────────────────────────────

def ask(query: str, use_rag: bool = True) -> dict:
    """
    Pipeline RAG complet :
    1. Récupère les produits pertinents (si use_rag=True)
    2. Formate le contexte
    3. Envoie au LLM
    4. Retourne la réponse + les sources

    Args:
        query    : question de l'utilisateur (FR, EN ou KR)
        use_rag  : False = répond sans contexte (pour comparaison)
    """
    if use_rag:
        # Étape 1 : Retrieval
        products = retrieve_products(query)
        context  = format_context(products, query)
        sources  = [f"{p['nom']} ({p['marque']})" for p in products]
    else:
        context  = "Aucun contexte fourni."
        sources  = []

    # Étape 2 : Augmentation + Generation
    chain    = prompt | llm
    response = chain.invoke({
        "context":  context,
        "question": query
    })

    return {
        "question": query,
        "answer":   response.content,
        "sources":  sources
    }

# ─────────────────────────────────────────────
# 8. Tests des 3 langues + code-switching
# ─────────────────────────────────────────────

if __name__ == "__main__":

    test_queries = [
        # Français
        "Je cherche une crème hydratante pour peau sèche, pas trop chère.",
        # Anglais
        "What is the best anti-aging serum you have?",
        # Coréen
        "건조한 피부에 좋은 크림 추천해 주세요.",
        # Code-switching FR + KR
        "Je veux un sérum. 피부 톤을 밝혀주는 제품 있나요?",
    ]

    print("\n" + "="*60)
    print("🧪 TESTS DU PIPELINE RAG MULTILINGUE")
    print("="*60)

    for query in test_queries:
        print(f"\n📝 Question : {query}")
        print("-"*50)

        result = ask(query)

        print(f"💬 Réponse :\n{result['answer']}")
        print(f"\n📦 Sources : {', '.join(result['sources'])}")
        print("="*60)
