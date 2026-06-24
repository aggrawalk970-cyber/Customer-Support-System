from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from app.agents.state import AgentState
from app.agents.llm import get_llm
from app.agents.tools import all_tools

SPECIALIST_SYSTEM_PROMPT = """You are the Specialist Agent for our Customer Support System.
You solve customer problems in billing, technical issues, and general support.
You have access to tools to:
1. Search the company knowledge base (FAQ, policies).
2. Retrieve order status by order ID.
3. Fetch user ticket history (long-term memory) by email.
4. Send emails to customers or supervisors.
5. Create support tickets.

Before solving a problem, ALWAYS check if you have the user's email:
- If you have the email, you MUST check their ticket history (using get_user_tickets) to see if they've reported similar problems before. If they have, mention it to show you remember them (e.g. "I see you reported X two weeks ago...").
- If they ask for information that you don't have, search the knowledge base.
- If they ask for order status, retrieve it using their order ID.

IMPORTANT: If you cannot solve the issue, or if the customer remains frustrated and explicitly asks for a human representative, say "I am routing you to the Escalation Agent for further assistance" and stop. This will trigger the escalation route.
"""

def specialist_node(state: AgentState) -> dict:
    messages = state["messages"]
    logs = state.get("logs", []) or []
    
    llm = get_llm()
    
    # We compile a sub-agent with tools using the LangGraph prebuilt react agent
    react_agent = create_react_agent(
        model=llm,
        tools=all_tools,
        state_modifier=SPECIALIST_SYSTEM_PROMPT
    )
    
    # Run the react agent on the history of messages
    response = react_agent.invoke({"messages": messages})
    new_messages = response["messages"]
    
    # The react agent appends its thoughts, tool calls, tool responses, and final response.
    # We slice only the newly appended messages to add back to the global state.
    newly_added = new_messages[len(messages):]
    
    # Check if final message triggers escalation
    final_content = new_messages[-1].content.lower()
    next_action = "end"
    if "escalat" in final_content or "human representative" in final_content:
        next_action = "escalation"
        
    log_msg = f"[Specialist Node]: Completed reasoning. Next Action determined: {next_action}."
    print(log_msg)
    
    return {
        "messages": newly_added,
        "next_action": next_action,
        "logs": logs + [log_msg]
    }
