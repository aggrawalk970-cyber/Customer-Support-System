"""
Interactive Demo CLI
====================
Showcases the full multi-agent system with beautiful Rich terminal output.
Demonstrates all three production debugging patterns.

Usage:
    python run_demo.py

Scenarios:
  1. Billing Query  — shows get_order_status tool use + long-term memory
  2. Technical Query — shows FAISS RAG retrieval with similarity scores
  3. Escalation + HITL — shows graph pause, supervisor approval, ticket creation
  4. RAG Debug Mode  — demonstrates how to diagnose retrieval vs generation failures
"""

import uuid
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

# Bootstrap the app (loads .env, creates DB tables, seeds data)
import app.main  # triggers @app.on_event("startup") side-effects via import

from langchain_core.messages import HumanMessage, AIMessage
from app.agents.graph import compiled_graph

console = Console()


# ── Visual Helpers ────────────────────────────────────────────────────────────

def header(title: str, subtitle: str = ""):
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]", justify="center")
    console.print()


def user_says(msg: str):
    console.print(Panel(
        f"[white]{msg}[/white]",
        title="[bold green]👤 User[/bold green]",
        border_style="green",
        padding=(0, 2),
    ))


def agent_says(msg: str, agent_name: str = "Agent"):
    icon = {"triage": "🔍", "billing_specialist": "💳", "technical_specialist": "🔧",
            "general_specialist": "📋", "escalation": "🚨"}.get(agent_name, "🤖")
    label = agent_name.replace("_", " ").title()
    console.print(Panel(
        f"[bright_white]{msg}[/bright_white]",
        title=f"[bold blue]{icon} {label}[/bold blue]",
        border_style="blue",
        padding=(0, 2),
    ))


def show_debug_info(values: dict):
    """Render the production debug panel — the core observability demo."""
    logs = values.get("logs", [])
    tool_calls = values.get("tool_calls_log", [])
    retrievals = values.get("retrieval_sources", [])

    # Agent Trajectory
    console.print(Panel(
        "\n".join(f"  • {log}" for log in logs) or "[dim]No logs[/dim]",
        title="[bold yellow]🗺  Agent Trajectory[/bold yellow]",
        border_style="yellow",
    ))

    # Tool Calls
    if tool_calls:
        table = Table(title="🔧 Tool Calls Log", box=box.ROUNDED, border_style="cyan")
        table.add_column("Tool", style="cyan", no_wrap=True)
        table.add_column("Inputs", style="white")
        table.add_column("Output (truncated)", style="green")
        table.add_column("Timestamp", style="dim")
        for tc in tool_calls:
            table.add_row(
                tc.get("tool_name", "?"),
                str(tc.get("inputs", {}))[:80],
                str(tc.get("output", ""))[:100],
                tc.get("timestamp", "")[:19],
            )
        console.print(table)

    # RAG Retrieval Sources
    if retrievals:
        for i, r in enumerate(retrievals):
            table = Table(
                title=f"📚 RAG Retrieval #{i+1} — Query: '{r.get('query', '')}'",
                box=box.ROUNDED,
                border_style="magenta",
            )
            table.add_column("Rank", style="dim", width=5)
            table.add_column("Score", style="bold")
            table.add_column("Chunk Preview (80 chars)", style="white")

            for rank, (chunk, score) in enumerate(
                zip(r.get("top_chunks", []), r.get("scores", [])), start=1
            ):
                score_color = "green" if score > 0.5 else ("yellow" if score > 0.3 else "red")
                table.add_row(
                    str(rank),
                    f"[{score_color}]{score:.3f}[/{score_color}]",
                    chunk[:80] + "...",
                )
            console.print(table)

            # The RAG Debugging Quadrant explanation
            scores_list = r.get("scores", [])
            if scores_list:
                max_score = max(scores_list)
                if max_score < 0.3:
                    console.print(Panel(
                        "[red]⚠ LOW RETRIEVAL SCORE (<0.3)[/red]\n"
                        "This is a [bold]RETRIEVAL failure[/bold]. The query didn't match relevant docs.\n"
                        "Fix: Improve chunking, add documents, or rewrite the search query.",
                        title="[bold red]🐛 RAG Debug Diagnosis[/bold red]",
                        border_style="red",
                    ))
                elif max_score < 0.5:
                    console.print(Panel(
                        "[yellow]⚠ MEDIUM RETRIEVAL SCORE (0.3–0.5)[/yellow]\n"
                        "Retrieval is marginal. If the answer is wrong, check BOTH retrieval and generation.\n"
                        "Inspect the 'output' field in Tool Calls Log — that's what the LLM received as context.",
                        title="[bold yellow]🐛 RAG Debug Diagnosis[/bold yellow]",
                        border_style="yellow",
                    ))
                else:
                    console.print(Panel(
                        "[green]✓ GOOD RETRIEVAL SCORE (>0.5)[/green]\n"
                        "If the answer is still wrong, it's a [bold]GENERATION failure[/bold] (hallucination).\n"
                        "Fix: Add citation requirements to prompt, lower temperature, or use a stronger model.",
                        title="[bold green]✓ RAG Debug Diagnosis[/bold green]",
                        border_style="green",
                    ))


def run_scenario(scenario_name: str, user_email: str, messages: list[str], auto_approve: bool = False):
    """Execute a full conversation scenario and display results."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    console.print(f"\n[bold magenta]Thread ID:[/bold magenta] [dim]{thread_id}[/dim]")

    for i, message in enumerate(messages):
        user_says(message)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Agents processing...[/bold blue]"),
            transient=True,
        ) as progress:
            progress.add_task("thinking", total=None)

            if i == 0:
                inputs = {
                    "messages": [HumanMessage(content=message)],
                    "user_email": user_email,
                    "intent": "general",
                    "next_action": "",
                    "active_agent": "",
                    "escalation_summary": "",
                    "approved_by_human": False,
                    "logs": [],
                    "tool_calls_log": [],
                    "retrieval_sources": [],
                }
                compiled_graph.invoke(inputs, config)
            else:
                current = compiled_graph.get_state(config)
                if current.next:
                    console.print(Panel(
                        "[bold yellow]⏸  Graph is PAUSED — waiting for human supervisor approval.[/bold yellow]\n"
                        f"Paused before node(s): {list(current.next)}",
                        border_style="yellow",
                    ))
                    if auto_approve:
                        time.sleep(0.5)
                        console.print("[bold green]✓ Supervisor approved escalation.[/bold green]")
                        compiled_graph.update_state(config, {"approved_by_human": True}, as_node="triage")
                        compiled_graph.invoke(None, config)
                    continue

                compiled_graph.update_state(config, {"messages": [HumanMessage(content=message)]})
                compiled_graph.invoke(None, config)

        final_state = compiled_graph.get_state(config)
        values = final_state.values

        # Show last AI message
        for msg in reversed(values.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                agent_says(msg.content, agent_name=values.get("active_agent", "agent"))
                break

        show_debug_info(values)

        if len(final_state.next) > 0 and not auto_approve:
            console.print(Panel(
                "[bold yellow]⏸  HITL PAUSE: Waiting for supervisor approval.[/bold yellow]\n"
                f"In a real system, a supervisor would call POST /api/approve.\n"
                f"Next node(s): {list(final_state.next)}",
                border_style="yellow",
            ))


# ── Main Demo ─────────────────────────────────────────────────────────────────

def main():
    console.print()
    console.print(Panel(
        Text.assemble(
            ("Multi-Agent Customer Support System\n", "bold cyan"),
            ("Production Agentic Pattern Demo\n\n", "bold white"),
            ("Triage → Specialist (ReAct) → Escalation / HITL\n", "dim"),
            ("LangGraph + FAISS RAG + SqliteSaver Checkpointing", "dim"),
        ),
        border_style="cyan",
        padding=(1, 4),
    ))

    scenarios = [
        {
            "name": "Scenario 1: Billing Query + Long-Term Memory",
            "subtitle": "Tests routing to billing_specialist, get_order_status tool, and ticket history lookup.",
            "email": "alice@example.com",
            "messages": ["Hi, I was double charged for my router order ORD12345. Can you help?"],
            "auto_approve": False,
        },
        {
            "name": "Scenario 2: Technical Query + FAISS RAG Debug",
            "subtitle": "Tests routing to technical_specialist and FAISS vector search with similarity scores.",
            "email": "bob@example.com",
            "messages": ["My internet keeps disconnecting. I've restarted my router twice already."],
            "auto_approve": False,
        },
        {
            "name": "Scenario 3: Escalation + Human-in-the-Loop Approval",
            "subtitle": "Tests HITL interrupt_before checkpoint, SqliteSaver persistence, and supervisor approval flow.",
            "email": "charlie@example.com",
            "messages": [
                "I am extremely frustrated. Nothing is working. I want a human supervisor NOW.",
                "(auto-approve triggered)",
            ],
            "auto_approve": True,
        },
    ]

    for scenario in scenarios:
        header(scenario["name"], scenario["subtitle"])
        run_scenario(
            scenario_name=scenario["name"],
            user_email=scenario["email"],
            messages=scenario["messages"],
            auto_approve=scenario.get("auto_approve", False),
        )
        console.print()
        time.sleep(1)

    header("Demo Complete", "All three agentic patterns demonstrated successfully.")
    console.print(Panel(
        "[bold green]✓ Resume Talking Points Demonstrated:[/bold green]\n\n"
        "  1. [cyan]Triage Agent[/cyan] — Structured Output (.with_structured_output()) instead of regex JSON parsing\n"
        "  2. [cyan]Specialist Agents[/cyan] — Three domain-scoped ReAct agents, pre-compiled at startup\n"
        "  3. [cyan]FAISS RAG[/cyan] — Vector similarity search with scores for retrieval debugging\n"
        "  4. [cyan]HITL[/cyan] — SqliteSaver checkpointing survives server restarts, interrupt_before gate\n"
        "  5. [cyan]Observability[/cyan] — debug_info in every API response (tool calls + retrieval sources)\n"
        "  6. [cyan]RAG Debugging Quadrant[/cyan] — Low score = retrieval bug; good score + wrong answer = generation bug",
        border_style="green",
        padding=(1, 2),
    ))


if __name__ == "__main__":
    main()
