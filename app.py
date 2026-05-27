"""
Cosmo·Universe — Chatbot RAG multilingue FR / KR / EN
Securite : rate limiting + sanitize input + confidentialite
+ correction comptage produits par marque
+ auto-indexation au demarrage si ChromaDB vide
"""

import os
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "cosmetiques_multilingue"
CSV_PATH        = "products.csv"
MAX_HISTORY     = 5
MAX_QUESTIONS   = 20
MAX_INPUT_LEN   = 500

CSS = """
<style>
.stApp { background-color: #FDF8F5; }
.block-container { padding-top: 3.5rem; max-width: 750px; }
h1 { font-family: Georgia, serif !important; font-weight: 500 !important; color: #1a1a1a !important; text-align: center; letter-spacing: 0.02em; }
.logo-sub { text-align: center; font-size: 12px; letter-spacing: 0.18em; color: #A8896C; text-transform: uppercase; margin-bottom: 0.25rem; }
.accent-bar { width: 36px; height: 3px; background: #D4A574; border-radius: 2px; margin: 0.5rem auto 1.5rem; }
.objective-box { background: white; border: 0.5px solid #F0E4DC; border-radius: 12px; padding: 14px 18px; font-size: 13px; color: #666; line-height: 1.7; margin-bottom: 0.5rem; text-align: center; }
.privacy-box { background: #FDF8F5; border: 0.5px solid #F0E4DC; border-radius: 8px; padding: 10px 18px; font-size: 11px; color: #aaa; line-height: 1.6; margin-bottom: 1.5rem; text-align: center; }
.hook-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 1.5rem; }
.hook-card { background: white; border: 0.5px solid #F0E4DC; border-radius: 12px; padding: 14px 12px; text-align: center; }
.hook-flag { font-size: 14px; font-weight: 500; color: #A8896C; margin-bottom: 6px; }
.hook-text { font-size: 11px; color: #888; line-height: 1.6; font-style: italic; }
div.stButton > button { background: white !important; border: 0.5px solid #E8D5C8 !important; border-radius: 20px !important; color: #555 !important; font-size: 12px !important; padding: 6px 14px !important; }
div.stButton > button:hover { background: #FDF0E8 !important; border-color: #D4A574 !important; }
.examples-label { font-size: 11px; font-weight: 500; color: #A8896C; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px; }
.lang-hint { text-align: center; font-size: 11px; color: #bbb; letter-spacing: 0.05em; margin-bottom: 6px; }
.stChatMessage { background: white !important; border: 0.5px solid #F0E4DC !important; border-radius: 12px !important; }
hr { border-color: #F0E4DC !important; }
</style>
"""

# ─────────────────────────────────────────────
# Vectorstore — chargement et auto-indexation
# ─────────────────────────────────────────────

@st.cache_resource
def load_vectorstore():
    """Charge ChromaDB — sans indexation."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

def index_products_if_needed():
    """
    Indexe products.csv dans ChromaDB si la collection est vide.
    Appelee depuis l interface Streamlit — pas depuis une fonction cachee.
    """
    vectorstore = load_vectorstore()

    if vectorstore._collection.count() > 0:
        return  # Deja indexe

    with st.spinner("Initialisation du catalogue... (premiere execution uniquement)"):
        df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

        documents = []
        metadatas = []
        ids       = []

        for i, row in df.iterrows():
            doc = (
                f"Produit: {row['nom']} | Brand: {row['marque']}\n"
                f"Categorie: {row['categorie']}\n"
                f"Prix: {row['prix_eur']}EUR\n"
                f"Caracteristiques: {row['caracteristiques']}\n"
                f"FR: {row['description_fr']}\n"
                f"KR: {row['description_kr']}\n"
                f"EN: {row['description_en']}"
            )
            documents.append(doc)
            metadatas.append({
                "nom":              str(row["nom"]),
                "marque":           str(row["marque"]),
                "categorie":        str(row["categorie"]),
                "prix_eur":         float(row["prix_eur"]),
                "caracteristiques": str(row["caracteristiques"]),
                "description_fr":   str(row["description_fr"]),
                "description_kr":   str(row["description_kr"]),
                "description_en":   str(row["description_en"]),
                "type_produit":     str(row["type_produit"]),
            })
            ids.append(f"product_{i}")

        for i in range(0, len(documents), 20):
            vectorstore.add_texts(
            texts=documents[i:i+20],
            metadatas=metadatas[i:i+20],
            ids=ids[i:i+20]
            )

@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=OPENAI_API_KEY
    )

# ─────────────────────────────────────────────
# Fonctions catalogue
# ─────────────────────────────────────────────

@st.cache_resource
def get_all_brands() -> list:
    vectorstore = load_vectorstore()
    results = vectorstore.get()
    return sorted(set(m["marque"] for m in results["metadatas"] if "marque" in m))

@st.cache_resource
def get_all_type_products() -> list:
    vectorstore = load_vectorstore()
    results = vectorstore.get()
    return sorted(set(m["type_produit"] for m in results["metadatas"] if "type_produit" in m))

@st.cache_resource
def get_all_categories() -> list:
    vectorstore = load_vectorstore()
    results = vectorstore.get()
    return sorted(set(m["categorie"] for m in results["metadatas"] if "categorie" in m))

@st.cache_resource
def get_price_range() -> dict:
    vectorstore = load_vectorstore()
    results = vectorstore.get()
    prices = [
        m["prix_eur"] for m in results["metadatas"]
        if "prix_eur" in m and isinstance(m["prix_eur"], (int, float))
    ]
    return {"min": min(prices) if prices else 0, "max": max(prices) if prices else 0}

def get_products_by_brand(brand: str) -> list:
    """Retourne TOUS les produits d une marque sans limite TOP_K."""
    vectorstore = load_vectorstore()
    results = vectorstore.get(where={"marque": brand})
    return results["metadatas"]

# ─────────────────────────────────────────────
# Securite — validation input
# ─────────────────────────────────────────────

def sanitize_input(query: str) -> str:
    query = query[:MAX_INPUT_LEN].strip()
    injection_patterns = [
        r"ignore (all |the |previous |above )?instructions?",
        r"forget (everything|all|previous)",
        r"you are now",
        r"act as",
        r"new instructions?",
        r"system prompt",
        r"disregard",
        r"모든 지시를 무시",
        r"너는 이제",
    ]
    for pattern in injection_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return ""
    return query

# ─────────────────────────────────────────────
# Dictionnaire types produits FR / KR / EN
# ─────────────────────────────────────────────

PRODUCT_TYPES = {
    "serum":         ["sérum", "serum", "세럼"],
    "creme":         ["crème", "creme", "크림"],
    "masque":        ["masque", "mask", "마스크"],
    "patch":         ["patch", "패치"],
    "huile":         ["huile", "oil", "오일"],
    "gel":           ["gel", "젤"],
    "lotion":        ["lotion", "lait", "로션", "유액"],
    "toner":         ["tonique", "toner", "토너"],
    "essence":       ["essence", "에센스"],
    "ampoule":       ["ampoule", "ampoul", "앰플"],
    "baume":         ["baume", "beurre", "밤", "버터"],
    "gommage":       ["gommage", "exfoliant", "scrub", "스크럽"],
    "nettoyant":     ["nettoyant", "démaquillant", "cleanser", "클렌저", "클렌징"],
    "maquillage":    ["maquillage", "makeup", "메이크업"],
    "fond_de_teint": ["fond de teint", "foundation", "파운데이션"],
    "levres":        ["lèvres", "lipstick", "글로스", "립"],
    "solaire":       ["solaire", "spf", "sunscreen", "선크림"],
    "parfum":        ["parfum", "fragrance", "향수"],
    "cheveux":       ["cheveux", "hair", "헤어"],
    "brume":         ["brume", "mist", "미스트"],
    "soin_yeux":     ["contour des yeux", "yeux", "eye", "아이"],
}

# ─────────────────────────────────────────────
# Detection de langue
# ─────────────────────────────────────────────

def detect_language(query: str, history: list = []) -> str:
    import unicodedata

    def remove_accents(text):
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

    # 1. Supprime les marques AVANT toute détection
    query_clean = query
    for brand in get_all_brands():
        query_clean = query_clean.replace(brand, "")
        query_clean = query_clean.replace(remove_accents(brand), "")
    query_clean = query_clean.strip()

    # 2. Compte les caractères par langue sur la query NETTOYEE
    korean_chars = sum(1 for c in query_clean if '\uAC00' <= c <= '\uD7A3')
    latin_chars  = sum(1 for c in query_clean if c.isalpha() and c.isascii())
    korean_words = len([w for w in query_clean.split() if any('\uAC00' <= c <= '\uD7A3' for c in w)])
    total_words  = len(query_clean.split())

    # 3. Si coréen présent — decide selon ratio de MOTS (pas de caractères)
    if korean_chars > 0:
        korean_ratio = korean_words / total_words if total_words > 0 else 0
        
        if korean_ratio >= 0.5:
            return "Korean"              # Majorité coréenne → Korean
        elif latin_chars == 0:
            return "Korean"              # Que du coréen → Korean
        else:
            # Coréen minoritaire → continue la détection FR/EN
            pass                         # ← on tombe dans la détection FR/EN ci-dessous

    # 4. Pas de coréen → détecte FR vs EN
    query_expanded = remove_accents(query_clean).lower()
    query_expanded = query_expanded.replace("i'm", "i am")
    query_expanded = query_expanded.replace("it's", "it is")
    query_expanded = query_expanded.replace("don't", "do not")

    fr_words = ["le", "la", "les", "un", "une", "des", "du", "je", "tu", "il",
                "nous", "vous", "est", "sont", "avec", "pour", "sur", "dans",
                "quel", "quelle", "comment", "combien", "produit", "donne",
                "liste", "avez", "avons", "cherche", "veux", "mon", "ma",
                "tout", "tous", "toute", "toutes", "qui", "que", "quoi"]

    en_words = ["the", "a", "an", "is", "are", "have", "how", "what", "which",
                "only", "do", "you", "can", "product", "under", "over", "best",
                "give", "list", "show", "tell", "find", "many", "brands", "not",
                "interested", "all", "looking", "want", "i", "in", "am", "me",
                "my", "this", "that", "with", "for", "about", "more", "less",
                "price", "brand", "recommend", "would", "like", "please",
                "could", "need", "search", "cheap", "expensive", "available", 
                "products", "of", "category", "categories", "type", "types",
                "show", "get", "give", "display", "anti", "aging", "skin", "care"]

    words    = query_expanded.split()
    fr_score = sum(1 for w in words if w in fr_words)
    en_score = sum(1 for w in words if w in en_words)

    # Phrase courte → langue du dernier message
    if len(words) < 3 and history:
        user_msgs = [m["content"] for m in history if m["role"] == "user"]
        if user_msgs:
            return detect_language(user_msgs[-1])

    # Mixte FR + EN
    if fr_score > 0 and en_score > 0:
        if en_score >= fr_score:
            return "English and French"
        else:
            return "French and English"

    if en_score > 0 and fr_score == 0:
        return "English"
    if fr_score > en_score:
        return "French"
    elif en_score > fr_score:
        return "English"
    else:
        return "French"  # vrai defaut — aucun mot reconnu

# ─────────────────────────────────────────────
# Detection marque dans la question
# ─────────────────────────────────────────────

def detect_brand_in_query(query: str, history: list) -> str:
    user_msgs = [m["content"] for m in history if m["role"] == "user"][-2:]
    full_context = " ".join(user_msgs + [query]).lower()
    for brand in get_all_brands():
        if brand.lower() in full_context:
            return brand
    return None

# ─────────────────────────────────────────────
# Extraction des filtres
# ─────────────────────────────────────────────

def extract_filters(query: str, history: list):
    user_msgs = [m["content"] for m in history if m["role"] == "user"][-3:]
    full_context = " ".join(user_msgs + [query]).lower()

    prix_filter   = None
    type_filter   = None
    marque_filter = None

    # "entre X et Y euros" / "X유로에서 Y유로"
    match = re.search(r"entre\s+(\d+)\s+et\s+(\d+)|(\d+)유로\s*에서\s*(\d+)유로", full_context)
    if match:
        low  = float(match.group(1) or match.group(3))
        high = float(match.group(2) or match.group(4))
        prix_filter = {"$and": [
            {"prix_eur": {"$gte": low}},
            {"prix_eur": {"$lte": high}}
        ]}

    # "moins de X" / "under X" / "en dessous de X" / "X유로 이하"
    match = re.search(r"moins de\s+(\d+)|under\s+(\d+)|en dessous de\s+(\d+)|(\d+)유로\s*이하", full_context)
    if match and not prix_filter:
        val = match.group(1) or match.group(2) or match.group(3) or match.group(4)
        prix_filter = {"prix_eur": {"$lte": float(val)}}

    # "plus de X" / "over X" / "X유로 이상"
    match = re.search(r"plus de\s+(\d+)|over\s+(\d+)|(\d+)유로\s*이상", full_context)
    if match and not prix_filter:
        val = match.group(1) or match.group(2) or match.group(3)
        prix_filter = {"prix_eur": {"$gte": float(val)}}

    # Filtre marque
    for brand in get_all_brands():
        if brand.lower() in full_context:
            marque_filter = {"marque": brand}
            break

    # Filtre type produit
    catalogue_questions = [
        "marque", "brand", "catégorie", "category",
        "liste", "list", "prix", "price", "브랜드", "카테고리",
        "combien", "how many", "몇 개", "nombre", "number", "total", "count"
    ]
    is_catalogue_question = any(kw in full_context for kw in catalogue_questions)

    if not is_catalogue_question:
        for type_key, keywords in PRODUCT_TYPES.items():
            if any(kw in full_context for kw in keywords):
                type_filter = {"type_produit": type_key}
                break

    # Combinaison filtres avec $and
    active_filters = []
    if prix_filter:
        if "$and" in prix_filter:
            active_filters.extend(prix_filter["$and"])
        else:
            active_filters.append(prix_filter)
    if type_filter:
        active_filters.append(type_filter)
    if marque_filter:
        active_filters.append(marque_filter)

    if len(active_filters) == 0:
        return None
    elif len(active_filters) == 1:
        return active_filters[0]
    else:
        return {"$and": active_filters}

# ─────────────────────────────────────────────
# TOP_K dynamique
# ─────────────────────────────────────────────

def get_top_k(query: str) -> int:
    large_keywords = ["liste", "list", "tous", "toutes", "all", "목록", "모든", "전체"]
    if any(kw in query.lower() for kw in large_keywords):
        return 10
    return 5

# ─────────────────────────────────────────────
# Formatage prix
# ─────────────────────────────────────────────

def format_price(price) -> str:
    try:
        return f"{float(price):,.2f} €".replace(",", " ").replace(".", ",")
    except:
        return f"{price} €"

# ─────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────

PROMPT_TEMPLATE = """
Tu es un assistant expert en cosmétiques pour une boutique mondiale haut de gamme.

CONNAISSANCE DU CATALOGUE :
- Marques disponibles : {brands}
- Types de produits : {type_products}
- Categories : {categories}
- Fourchette de prix : {price_min} a {price_max}
- Nombre TOTAL de produits de la marque demandee : {brand_total}

INSTRUCTIONS :
- La langue detectee de la question est : {detected_lang}
- Tu DOIS repondre en {detected_lang}, sans exception.
- REGLE ABSOLUE :
    * Question en francais   -> reponse en FRANCAIS UNIQUEMENT
    * Question en coreen     -> reponse en COREEN UNIQUEMENT
    * Question en anglais    -> reponse en ANGLAIS UNIQUEMENT
    * Question melangee FR+KR -> reponds dans les deux langues naturellement
    * Question melangee EN+KR -> reponds dans les deux langues naturellement
    * Question melangee FR+EN -> reponds dans les deux langues naturellement
- Ne te laisse PAS influencer par la langue des descriptions produits dans le contexte.
- La langue de ta reponse = la langue de la QUESTION, toujours.
- Tiens compte de l historique pour comprendre le contexte.
- Si l utilisateur dit "le moins cher" ou "detail" sans preciser le produit,
  refere-toi a l historique pour comprendre de quel produit il parle.
- Pour les questions sur les marques, types, categories ou prix -> utilise
  la CONNAISSANCE DU CATALOGUE ci-dessus.
- Pour le NOMBRE de produits d une marque -> utilise TOUJOURS "brand_total" ci-dessus,
  jamais le nombre de produits dans PRODUITS DISPONIBLES.
- Base ta reponse EXCLUSIVEMENT sur les produits listes dans PRODUITS DISPONIBLES.
- Si un produit n apparait pas dans PRODUITS DISPONIBLES, il n existe pas — ne l invente JAMAIS.
- Si aucun produit ne correspond, dis CLAIREMENT que tu n as pas ce produit.
- Ne complete JAMAIS avec des produits imaginaires ou issus de tes connaissances generales.
- Mentionne toujours le nom du produit, la marque et le prix formate en euros.
- Sois chaleureux et professionnel, comme un conseiller en boutique de luxe.

HISTORIQUE (derniers {max_history} echanges) :
{history}

PRODUITS DISPONIBLES ({nb_products} produits affiches sur {brand_total} au total) :
{context}

QUESTION : {question}

REPONSE :
"""

# ─────────────────────────────────────────────
# Fonctions RAG
# ─────────────────────────────────────────────

def build_enriched_query(query: str, history: list) -> str:
    words = query.lower().split()
    specific_keywords = [
        "marque", "brand", "브랜드",
        "prix", "price", "가격",
        "liste", "list", "목록",
        "catégorie", "category", "카테고리",
        "produit", "product", "제품"
    ]
    if any(kw in query.lower() for kw in specific_keywords):
        return query
    if len(words) < 5:
        user_msgs = [m["content"] for m in history if m["role"] == "user"][-3:]
        if user_msgs:
            return " ".join(user_msgs) + " " + query
    return query

def retrieve_products(query: str, history: list) -> tuple:
    count_keywords = ["combien", "how many", "몇 개", "nombre", "number", "total", "count"]
    is_count_question = any(kw in query.lower() for kw in count_keywords)
    brand = detect_brand_in_query(query, history)

    if brand and is_count_question:
        all_products = get_products_by_brand(brand)
        return all_products[:10], len(all_products)

    vectorstore = load_vectorstore()
    top_k       = get_top_k(query)
    enriched    = build_enriched_query(query, history)
    filters     = extract_filters(query, history)
    results     = vectorstore.similarity_search(enriched, k=top_k, filter=filters)
    products    = [doc.metadata for doc in results]

    if brand:
        total = len(get_products_by_brand(brand))
    else:
        total = len(products)

    return products, total

def format_context(products: list) -> str:
    parts = []
    for i, p in enumerate(products, 1):
        parts.append(
            f"Produit {i}:\n"
            f"  Nom: {p.get('nom', 'N/A')}\n"
            f"  Marque: {p.get('marque', 'N/A')}\n"
            f"  Type: {p.get('type_produit', 'N/A')}\n"
            f"  Categorie: {p.get('categorie', 'N/A')}\n"
            f"  Prix: {format_price(p.get('prix_eur', 0))}\n"
            f"  Caracteristiques: {p.get('caracteristiques', 'N/A')}\n"
            f"  Description FR: {p.get('description_fr', 'N/A')}\n"
            f"  Description KR: {p.get('description_kr', 'N/A')}\n"
            f"  Description EN: {p.get('description_en', 'N/A')}\n"
        )
    return "\n".join(parts)

def format_history(history: list) -> str:
    recent = history[-(MAX_HISTORY * 2):]
    if not recent:
        return "Aucun historique."
    lines = []
    for msg in recent:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content'][:300]}")
    return "\n".join(lines)

def format_sources(products: list) -> list:
    return [f"{p['nom']} ({p['marque']}) — {format_price(p['prix_eur'])}" for p in products]

def ask(query: str, history: list) -> dict:
    products, brand_total = retrieve_products(query, history)
    context       = format_context(products)
    sources       = format_sources(products)
    hist_text     = format_history(history)
    detected_lang = detect_language(query, history)
    price_range   = get_price_range()

    prompt   = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain    = prompt | load_llm()    
    response = chain.invoke({
        "context":       context,
        "question":      query,
        "history":       hist_text,
        "max_history":   MAX_HISTORY,
        "detected_lang": detected_lang,
        "brands":        ", ".join(get_all_brands()),
        "type_products": ", ".join(get_all_type_products()),
        "categories":    ", ".join(get_all_categories()),
        "price_min":     format_price(price_range["min"]),
        "price_max":     format_price(price_range["max"]),
        "nb_products":   len(products),
        "brand_total":   brand_total,
    })
    return {"answer": response.content, "sources": sources}

# ─────────────────────────────────────────────
# Interface Streamlit
# ─────────────────────────────────────────────

st.set_page_config(page_title="Cosmo Universe", page_icon="💄", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown('<div class="logo-sub">Beauté · 뷰티 · Beauty</div>', unsafe_allow_html=True)
st.title("💄 Cosmo·Universe")
st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="objective-box">
✨ Vous cherchez un produit ? Nous avons la réponse.<br>
Des milliers de cosmétiques référencés, une IA multilingue pour vous guider.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="privacy-box">
🔒 Vos questions sont traitées de manière anonyme et ne sont pas conservées après votre session.<br>
Aucune donnée personnelle n est collectée. · 개인 데이터는 수집되지 않습니다. · No personal data is collected.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hook-grid">
    <div class="hook-card">
        <div class="hook-flag">🇫🇷 Français</div>
        <div class="hook-text">Tous les cosmétiques du monde. Toutes les langues. Une seule réponse.</div>
    </div>
    <div class="hook-card">
        <div class="hook-flag">🇰🇷 한국어</div>
        <div class="hook-text">세상의 모든 뷰티, 모든 언어, 하나의 답.</div>
    </div>
    <div class="hook-card">
        <div class="hook-flag">🇬🇧 English</div>
        <div class="hook-text">All the cosmetics in the world. Every language. One answer.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Auto-indexation si ChromaDB vide
index_products_if_needed()

st.divider()

# Initialisation session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "request_count" not in st.session_state:
    st.session_state.request_count = 0

# Sidebar
# Sidebar — placeholder mis a jour apres chaque question
with st.sidebar:
    st.markdown("## 💄 Cosmo·Universe")
    st.divider()

    st.markdown("**💬 Questions**")
    counter_placeholder = st.empty()  # ← placeholder
    counter_placeholder.progress(st.session_state.request_count / MAX_QUESTIONS)
    counter_caption = st.empty()
    counter_caption.caption(f"{st.session_state.request_count} / {MAX_QUESTIONS} questions utilisées")

    st.divider()

    if st.button("🔄 Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.request_count = 0
        st.rerun()

    st.divider()

    st.markdown("**🌍 Langues**")
    st.caption("🇫🇷 Français · 🇰🇷 한국어 · 🇬🇧 English")
    st.caption("Posez vos questions dans la langue de votre choix.")

# Boutons exemples
st.markdown('<div class="examples-label">💡 Exemples de questions</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🇫🇷 Crème pour peau sèche"):
        st.session_state.example = "Je cherche une crème hydratante pour peau sèche."
with col2:
    if st.button("🇰🇷 추천 세럼"):
        st.session_state.example = "피부 톤을 밝혀주는 세럼 추천해 주세요."
with col3:
    if st.button("🇬🇧 Best anti-aging serum"):
        st.session_state.example = "What is the best anti-aging serum you have?"

st.divider()

# Affichage historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📦 Produits consultés"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

# Limite atteinte
if st.session_state.request_count >= MAX_QUESTIONS:
    st.warning(
        f"Vous avez atteint la limite de {MAX_QUESTIONS} questions pour cette session. "
        "Cliquez sur Nouvelle conversation pour recommencer."
    )
    st.stop()

# Input
default_input = st.session_state.pop("example", "")
st.markdown('<div class="lang-hint">🇫🇷 Français · 🇰🇷 한국어 · 🇬🇧 English</div>', unsafe_allow_html=True)
query = st.chat_input("Recherchez-vous un produit particulier ?")

if default_input and not query:
    query = default_input

# Traitement
if query:
    query_clean = sanitize_input(query)

    if not query_clean:
        st.warning("Votre message contient des éléments non autorisés.")
        st.stop()

    st.session_state.request_count += 1  # ← incrémenter EN PREMIER

    # Mettre a jour le compteur sidebar immédiatement
    counter_placeholder.progress(st.session_state.request_count / MAX_QUESTIONS)
    counter_caption.caption(f"{st.session_state.request_count} / {MAX_QUESTIONS} questions utilisées")

    st.session_state.messages.append({"role": "user", "content": query_clean})
    
    with st.chat_message("user"):
        st.markdown(query_clean)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            history = st.session_state.messages[:-1]
            result  = ask(query_clean, history)
        st.markdown(result["answer"])
        with st.expander("📦 Produits consultés"):
            for source in result["sources"]:
                st.markdown(f"- {source}")

    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
