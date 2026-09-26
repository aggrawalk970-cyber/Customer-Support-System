import tiktoken
from langchain_core.messages import RemoveMessage, trim_messages, BaseMessage
from app.agents.state import AgentState
from app.core.config import settings

def count_tokens(messages: list[BaseMessage]) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return sum(len(enc.encode(m.content)) for m in messages if m.content)

def trim_memory_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {}
        
    retained_messages = trim_messages(
        messages,
        max_tokens=settings.MAX_TOKENS,
        strategy="last",
        token_counter=count_tokens,
        include_system=True,
        allow_partial=False
    )
    
    retained_ids = {m.id for m in retained_messages if m.id}
    messages_to_remove = [RemoveMessage(id=m.id) for m in messages if m.id and m.id not in retained_ids]
    
    if messages_to_remove:
        log_msg = f"[Memory Node]: Trimmed {len(messages_to_remove)} old messages to stay under {settings.MAX_TOKENS} tokens."
        print(log_msg)
        return {
            "messages": messages_to_remove,
            "logs": [log_msg]
        }
        
    return {}
