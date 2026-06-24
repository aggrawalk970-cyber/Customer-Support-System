from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.triage import triage_node
from app.agents.specialist import specialist_node
from app.agents.escalation import escalation_node

# Router functions for conditional transitions
def route_after_triage(state: AgentState):
    if state.get("next_action") == "escalation":
        return "escalation"
    return "specialist"

def route_after_specialist(state: AgentState):
    if state.get("next_action") == "escalation":
        return "escalation"
    return END

# 1. Instantiate the State Graph
workflow = StateGraph(AgentState)

# 2. Register agent nodes
workflow.add_node("triage", triage_node)
workflow.add_node("specialist", specialist_node)
workflow.add_node("escalation", escalation_node)

# 3. Configure flow paths
workflow.set_entry_point("triage")

workflow.add_conditional_edges(
    "triage",
    route_after_triage,
    {
        "specialist": "specialist",
        "escalation": "escalation"
    }
)

workflow.add_conditional_edges(
    "specialist",
    route_after_specialist,
    {
        "escalation": "escalation",
        END: END
    }
)

workflow.add_edge("escalation", END)

# 4. Memory checkpointer for tracking conversations (Short-term memory)
memory = MemorySaver()

# 5. Compile the graph with a Human-in-the-loop block before escalation runs
compiled_graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["escalation"]
)
