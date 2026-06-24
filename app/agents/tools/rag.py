from langchain_core.tools import tool

# Mock company knowledge base documents
MOCK_DOCUMENTS = [
    {
        "title": "Refund and Billing Policy",
        "content": "Refunds are processed within 14 days of purchase. To initiate a refund, please provide your order ID. If the purchase was made more than 14 days ago, refunds are generally not allowed unless there is a billing error. Billing disputes should be escalated or sent directly to billing@company.com."
    },
    {
        "title": "Shipping and Deliveries",
        "content": "We offer standard shipping (3-5 business days) and express shipping (1-2 business days). Orders are processed within 24 hours of payment confirmation. Once shipped, customers receive a tracking ID to check status. Canceled orders are fully refunded if they have not yet shipped."
    },
    {
        "title": "Technical Troubleshooting Guide",
        "content": "For internet/router connection issues: restart your device, hold the physical reset button for 10 seconds, or verify that your DSL/Fiber cables are connected. For software glitches: update the client application, clear your browser cache, or re-install the application. If issues persist, contact technical support."
    },
    {
        "title": "General Contact and Support Hours",
        "content": "Customer service is open Monday through Friday, 9:00 AM to 5:00 PM EST. Our support team can be reached via support@company.com or by filing a ticket through this portal. We aim to reply to all queries within 24 hours."
    }
]

@tool
def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base (RAG docs) for answers to FAQs, billing rules, technical troubleshooting steps, and support policies.
    
    Args:
        query: The search term or question to query the database.
    """
    # Simple token-based keyword search to simulate standard vector RAG ranking
    query_words = set(w.strip(",.?!()\"'") for w in query.lower().split() if len(w) > 2)
    scored_docs = []
    
    for doc in MOCK_DOCUMENTS:
        text_pool = (doc["content"] + " " + doc["title"]).lower()
        score = 0
        for word in query_words:
            if word in text_pool:
                # Give higher weight to matches in the title
                if word in doc["title"].lower():
                    score += 3
                else:
                    score += 1
        if score > 0:
            scored_docs.append((score, doc))
            
    # Sort docs by matching score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    if not scored_docs:
        titles = ", ".join([f"'{d['title']}'" for d in MOCK_DOCUMENTS])
        return f"No articles matched your query. Available articles include: {titles}."
    
    results = []
    # Return top 2 matching articles
    for score, doc in scored_docs[:2]:
        results.append(f"--- Article: {doc['title']} ---\n{doc['content']}")
        
    return "\n\n".join(results)
