"""
RAG Tool — FAISS Vector Search with SentenceTransformers
──────────────────────────────────────────────────────────
Replaces the keyword token-matching approach with real semantic vector search.

Production Debugging Pattern (RAG Quadrant):
  - We return similarity scores alongside retrieved chunks.
  - If a wrong answer occurs, you check:
    (A) Were similarity scores low (< 0.4)? → Retrieval Bug (bad query or index).
    (B) Were scores high but answer wrong? → Generation Bug (hallucination or context ignored).
  - This diagnostic is surfaced in the API's `debug_info.retrieval_sources` field.

Interview Talking Point:
  This pattern separates retrieval failures from generation failures — a critical
  skill for debugging production RAG pipelines. Most teams skip this and spend hours
  debugging the wrong component.
"""
import datetime
import numpy as np
from langchain_core.tools import tool

# ── Lazy-loaded singletons (expensive to load, done once at first call) ────────
_faiss_index = None
_embedder = None
_documents: list[dict] = []
_last_retrieval = {}

def get_last_retrieval() -> dict:
    global _last_retrieval
    return _last_retrieval

# ── Mock Company Knowledge Base Documents ─────────────────────────────────────
MOCK_DOCUMENTS = [
    {
        "title": "Refund and Billing Policy",
        "content": (
            "Refunds are processed within 14 days of purchase. To initiate a refund, please "
            "provide your order ID. If the purchase was made more than 14 days ago, refunds are "
            "generally not allowed unless there is a billing error. Billing disputes should be "
            "escalated or sent directly to billing@company.com."
        ),
    },
    {
        "title": "Shipping and Deliveries",
        "content": (
            "We offer standard shipping (3-5 business days) and express shipping (1-2 business days). "
            "Orders are processed within 24 hours of payment confirmation. Once shipped, customers "
            "receive a tracking ID to check status. Canceled orders are fully refunded if they have "
            "not yet shipped."
        ),
    },
    {
        "title": "Technical Troubleshooting Guide",
        "content": (
            "For internet/router connection issues: restart your device, hold the physical reset "
            "button for 10 seconds, or verify that your DSL/Fiber cables are connected. For software "
            "glitches: update the client application, clear your browser cache, or re-install the "
            "application. If issues persist, contact technical support."
        ),
    },
    {
        "title": "General Contact and Support Hours",
        "content": (
            "Customer service is open Monday through Friday, 9:00 AM to 5:00 PM EST. "
            "Our support team can be reached via support@company.com or by filing a ticket through "
            "this portal. We aim to reply to all queries within 24 hours."
        ),
    },
    {
        "title": "Subscription Plans and Pricing",
        "content": (
            "We offer three subscription tiers: Basic ($9.99/month), Pro ($24.99/month), and "
            "Enterprise (custom pricing). Annual billing provides a 20% discount. Upgrades take "
            "effect immediately and the price difference is prorated. Downgrades take effect at "
            "the next billing cycle."
        ),
    },
    {
        "title": "Password and Account Recovery",
        "content": (
            "To reset your password, click 'Forgot Password' on the login page and enter your "
            "registered email. A reset link will be sent within 5 minutes. If you don't receive it, "
            "check your spam folder. For locked accounts, contact support@company.com with your "
            "account ID for manual unlock within 24 hours."
        ),
    },
]


def _get_faiss_index():
    """Lazy-load and build the FAISS index + SentenceTransformer model on first call.

    This is intentionally lazy to avoid expensive model loading at import time,
    keeping FastAPI startup fast.
    """
    global _faiss_index, _embedder, _documents

    if _faiss_index is not None:
        return _faiss_index, _embedder, _documents

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        from app.core.config import settings

        print("[RAG]: Loading SentenceTransformer model for FAISS indexing...")
        _embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

        # Embed all documents and build the FAISS flat L2 index
        texts = [f"{doc['title']}. {doc['content']}" for doc in MOCK_DOCUMENTS]
        embeddings = _embedder.encode(texts, normalize_embeddings=True).astype("float32")

        # Inner Product on normalized vectors = cosine similarity
        dim = embeddings.shape[1]
        _faiss_index = faiss.IndexFlatIP(dim)
        _faiss_index.add(embeddings)
        _documents = MOCK_DOCUMENTS

        print(f"[RAG]: FAISS index built. {len(_documents)} documents indexed (dim={dim}).")

    except ImportError as e:
        print(f"[RAG]: FAISS/SentenceTransformers not installed ({e}). Falling back to keyword search.")
        _faiss_index = "keyword_fallback"
        _documents = MOCK_DOCUMENTS

    return _faiss_index, _embedder, _documents


def _keyword_fallback_search(query: str, top_k: int = 2) -> list[dict]:
    """Keyword-based fallback search when FAISS is unavailable."""
    query_words = set(w.strip(",.?!()\"'") for w in query.lower().split() if len(w) > 2)
    scored = []
    for doc in MOCK_DOCUMENTS:
        text_pool = (doc["content"] + " " + doc["title"]).lower()
        score = sum(3 if w in doc["title"].lower() else 1 for w in query_words if w in text_pool)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"document": doc, "score": round(score / 10.0, 3)}
        for score, doc in scored[:top_k]
    ]


def _search_personal_memory_faiss(query: str, email: str, embedder, top_k: int = 1) -> list:
    from app.db.session import SessionLocal
    from app.db.models import Ticket, UserProfile
    import faiss
    
    db = SessionLocal()
    docs = []
    try:
        profile = db.query(UserProfile).filter(UserProfile.user_email == email).first()
        if profile and profile.context_summary:
            docs.append({"title": "User Profile Summary", "content": profile.context_summary})
            
        tickets = db.query(Ticket).filter(Ticket.user_email == email, Ticket.conversation_summary != None).all()
        for t in tickets:
            docs.append({"title": f"Past Ticket #{t.id}", "content": t.conversation_summary})
            
        if not docs:
            return []
            
        texts = [f"{doc['title']}. {doc['content']}" for doc in docs]
        embeddings = embedder.encode(texts, normalize_embeddings=True).astype("float32")
        
        dim = embeddings.shape[1]
        temp_index = faiss.IndexFlatIP(dim)
        temp_index.add(embeddings)
        
        query_embedding = embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = temp_index.search(query_embedding, min(top_k, len(docs)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > 0.3: # threshold
                results.append({
                    "document": docs[idx],
                    "score": float(round(score, 4)),
                })
        return results
    except Exception as e:
        print(f"[RAG] Personal memory search error: {e}")
        return []
    finally:
        db.close()


@tool
def search_knowledge_base(query: str, user_email: str = "") -> str:
    """Search the company knowledge base using semantic vector similarity (FAISS + SentenceTransformers).
    If user_email is provided, it also retrieves relevant past context (Semantic memory retrieval).

    Returns relevant FAQ articles, policy documents, and relevant past user tickets.
    Includes similarity scores for RAG pipeline diagnostics.

    Args:
        query: The customer's question or the topic to search for.
        user_email: The customer's email to search personal history.
    """
    index, embedder, documents = _get_faiss_index()
    top_k = 2

    results_with_scores = []

    if index == "keyword_fallback" or index is None:
        results_with_scores = _keyword_fallback_search(query, top_k=top_k)
    else:
        # Semantic vector search using FAISS
        query_embedding = embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = index.search(query_embedding, top_k)

        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results_with_scores.append({
                    "document": documents[idx],
                    "score": float(round(score, 4)),  # Cosine similarity (0.0 – 1.0)
                })
                
        if user_email:
            personal_results = _search_personal_memory_faiss(query, user_email, embedder, top_k=1)
            results_with_scores.extend(personal_results)
            # sort again by score
            results_with_scores.sort(key=lambda x: x["score"], reverse=True)

    if not results_with_scores:
        titles = ", ".join([f"'{d['title']}'" for d in MOCK_DOCUMENTS])
        return f"No relevant articles found for your query. Available topics: {titles}."

    global _last_retrieval
    _last_retrieval = {
        "query": query,
        "scores": [r["score"] for r in results_with_scores],
        "top_chunks": [r["document"]["title"] for r in results_with_scores]
    }

    # Format output for the LLM agent — include scores for debugging visibility
    formatted = []
    for result in results_with_scores:
        doc = result["document"]
        score = result["score"]
        quality = "HIGH" if score >= 0.6 else "MEDIUM" if score >= 0.35 else "LOW"
        formatted.append(
            f"--- [{quality} relevance | similarity={score}] {doc['title']} ---\n{doc['content']}"
        )

    return "\n\n".join(formatted)
