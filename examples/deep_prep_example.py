"""
Example: Deep Preparation Orchestrator

This example demonstrates how to use the orchestrator agent to run
debate preparation in either "lite" mode (research only) or "full" mode
(research + constructive speech).

The orchestrator uses a ReAct pattern with:
- Todo list management for progress tracking
- Subgraph tools for topic_research and constructive_speech
- Automatic decision-making about which tools to call

Prerequisites:
- COMET_API_KEY environment variable must be set
- All prompts must be uploaded to Opik platform
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler
from rich.markdown import Markdown

from llm_debate_assistant.core.initialization import init_app
from llm_debate_assistant.agents.deep_preparation import (
    run_preparation,
    create_orchestrator,
    create_initial_state,
    create_filesystem,
    PrepMode,
)

# Setup logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger(__name__)

console = Console()


def display_messages(messages: list, max_display: int = 10):
    """Display conversation messages in a formatted way."""
    console.print("\n[bold cyan]Conversation History:[/bold cyan]")

    # Show last N messages
    display_msgs = messages[-max_display:] if len(messages) > max_display else messages
    if len(messages) > max_display:
        console.print(f"[dim](Showing last {max_display} of {len(messages)} messages)[/dim]\n")

    for i, msg in enumerate(display_msgs):
        msg_type = type(msg).__name__

        # Color and emoji based on message type
        if msg_type == "HumanMessage":
            emoji = "👤"
            color = "green"
            label = "User"
        elif msg_type == "AIMessage":
            emoji = "🤖"
            color = "blue"
            label = "Agent"
        elif msg_type == "ToolMessage":
            emoji = "🔧"
            color = "yellow"
            label = f"Tool ({getattr(msg, 'name', 'unknown')})"
        else:
            emoji = "📝"
            color = "white"
            label = msg_type

        # Get content preview
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            preview = content[:200] + "..." if len(content) > 200 else content
        else:
            preview = str(content)[:200]

        # Check for tool calls
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            tool_names = [tc.get("name", "?") for tc in tool_calls]
            console.print(f"{emoji} [{color}]{label}[/{color}]: [dim]Calling tools: {tool_names}[/dim]")
        elif preview.strip():
            console.print(f"{emoji} [{color}]{label}[/{color}]: {preview}")
        else:
            console.print(f"{emoji} [{color}]{label}[/{color}]: [dim](empty)[/dim]")


def display_stage_status(stage_status: dict):
    """Display the stage status dashboard."""
    if not stage_status:
        console.print("[dim]No stage status recorded[/dim]")
        return

    table = Table(title="Stage Status")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="magenta")

    status_emoji = {
        "completed": "✅",
        "in_progress": "🔄",
        "not_started": "⏸️",
        "failed": "❌",
    }

    for stage, status in stage_status.items():
        emoji = status_emoji.get(status, "❓")
        table.add_row(stage, f"{emoji} {status}")

    console.print(table)


def display_artifact_paths(artifact_paths: dict):
    """Display saved artifact paths."""
    if not artifact_paths:
        console.print("[dim]No artifacts recorded[/dim]")
        return

    console.print("\n[bold cyan]Saved Artifacts:[/bold cyan]")
    for name, path in artifact_paths.items():
        console.print(f"  📄 [bold]{name}[/bold]: [dim]{path}[/dim]")


async def run_example(
    topic: str,
    side: Literal["正方", "反方"],
    mode: PrepMode,
):
    """Run the orchestrator example with given parameters."""

    # Header
    mode_label = "LITE (Research Only)" if mode == "lite" else "FULL (Research + Speech)"
    console.print(
        Panel.fit(
            f"[bold]Topic:[/bold] {topic}\n"
            f"[bold]Side:[/bold] {side}\n"
            f"[bold]Mode:[/bold] {mode_label}",
            title="🎯 Deep Preparation Orchestrator",
            border_style="blue",
        )
    )

    # Create filesystem for this run
    console.print("\n[dim]Setting up filesystem...[/dim]")
    filesystem, session_path, cache_found = create_filesystem(
        filesystem_type="disk",
        topic=topic,
        side=side,
        segment="orchestrator",
        use_cache=True,
    )

    if cache_found:
        console.print(f"[green]✓ Found existing session[/green]")
    else:
        console.print(f"[blue]⚡ Creating new session[/blue]")

    if session_path:
        relative_path = Path(session_path).relative_to(Path.cwd())
        console.print(f"[dim]Session path: {relative_path}[/dim]")

    # Create orchestrator
    console.print("\n[bold cyan]Creating orchestrator graph...[/bold cyan]")
    graph = create_orchestrator(mode=mode, filesystem=filesystem)
    app = graph.compile()

    # Create initial state
    initial_state = create_initial_state(
        topic=topic,
        side=side,
        mode=mode,
    )

    # Display workflow info
    console.print(
        Panel(
            "[bold]Workflow:[/bold]\n"
            "1. Agent receives task and creates todo list\n"
            "2. Agent adds tasks and marks them in progress\n"
            "3. Agent calls tools (topic_research, constructive_speech)\n"
            "4. Agent marks tasks complete and reports results\n\n"
            "[dim]The agent autonomously decides when to call each tool[/dim]",
            title="🔄 ReAct Agent Loop",
            border_style="cyan",
        )
    )

    # Run the orchestrator
    console.print("\n[bold green]Starting orchestrator...[/bold green]")
    console.print("[dim](This may take several minutes depending on mode)[/dim]\n")

    start_time = time.time()

    try:
        # Stream events for real-time progress
        final_state = None
        step_count = 0

        async for event in app.astream(initial_state):
            step_count += 1
            for node_name, node_output in event.items():
                if node_name == "agent":
                    # Check if agent made tool calls
                    messages = node_output.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        tool_calls = getattr(last_msg, "tool_calls", None)
                        if tool_calls:
                            tools = [tc.get("name", "?") for tc in tool_calls]
                            console.print(f"[blue]🤖 Agent calling:[/blue] {', '.join(tools)}")
                        elif hasattr(last_msg, "content") and last_msg.content:
                            # Agent responding without tools (likely final response)
                            preview = last_msg.content[:100] + "..." if len(last_msg.content) > 100 else last_msg.content
                            console.print(f"[blue]�� Agent:[/blue] {preview}")
                elif node_name == "tools":
                    # Tool execution completed
                    messages = node_output.get("messages", [])
                    if messages:
                        for msg in messages:
                            tool_name = getattr(msg, "name", "unknown")
                            content = getattr(msg, "content", "")
                            # Show first line of tool output
                            first_line = content.split("\n")[0][:80] if content else "(no output)"
                            console.print(f"[yellow]🔧 {tool_name}:[/yellow] {first_line}")

                # Update final state
                if isinstance(node_output, dict):
                    if final_state is None:
                        final_state = dict(initial_state)
                    final_state.update(node_output)

        execution_time = time.time() - start_time

        # Success header
        console.print("\n" + "=" * 80)
        console.print(
            Panel.fit(
                f"[bold green]✓ Orchestrator completed in {execution_time:.1f}s[/bold green]\n"
                f"[dim]Total steps: {step_count}[/dim]",
                border_style="green",
            )
        )

        # Display results
        if final_state:
            # Stage status
            console.print("\n")
            display_stage_status(final_state.get("stage_status", {}))

            # Artifact paths
            display_artifact_paths(final_state.get("artifact_paths", {}))

            # Todo list summary (if available)
            todo_list = final_state.get("todo_list")
            if todo_list:
                console.print("\n[bold cyan]Final Todo List:[/bold cyan]")
                console.print(todo_list.format_summary())

            # Conversation history
            messages = final_state.get("messages", [])
            if messages:
                display_messages(messages)

            # Final agent response
            if messages:
                last_ai_msg = None
                for msg in reversed(messages):
                    if type(msg).__name__ == "AIMessage" and not getattr(msg, "tool_calls", None):
                        last_ai_msg = msg
                        break

                if last_ai_msg and last_ai_msg.content:
                    console.print("\n")
                    console.print(
                        Panel(
                            Markdown(last_ai_msg.content),
                            title="📋 Final Report",
                            border_style="green",
                        )
                    )

        # Session info
        if session_path:
            relative_path = Path(session_path).relative_to(Path.cwd())
            console.print(f"\n[dim]All artifacts saved to: {relative_path}[/dim]")

    except Exception as e:
        execution_time = time.time() - start_time
        console.print(f"\n[red]✗ Error after {execution_time:.1f}s: {e}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")


async def main():
    """Run orchestrator examples."""

    console.print("[bold yellow]" + "=" * 80)
    console.print("[bold yellow]Deep Preparation Orchestrator Examples")
    console.print("[bold yellow]" + "=" * 80 + "\n")

    # Initialize application
    console.print("[cyan]Initializing application...[/cyan]")
    try:
        await init_app()
        console.print("[green]✓ Application initialized[/green]\n")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not fully initialize: {e}[/yellow]")
        console.print("[yellow]Continuing anyway...[/yellow]\n")

    # Example topic
    topic = "应该强制要求大型科技公司开源其核心算法"
    side = "正方"

    # Ask user which mode to run
    console.print("[bold]Select mode:[/bold]")
    console.print("  1. [cyan]lite[/cyan] - Topic research only (faster, ~2-3 min)")
    console.print("  2. [magenta]full[/magenta] - Research + Constructive speech (~5-8 min)")
    console.print()

    user_input = input("Enter choice (1/2) [default: 1]: ").strip()

    if user_input == "2":
        mode: PrepMode = "full"
    else:
        mode: PrepMode = "lite"

    console.print(f"\n[bold]Running in {mode.upper()} mode...[/bold]\n")

    # Run the example
    await run_example(topic=topic, side=side, mode=mode)

    console.print("\n[bold green]✨ Example complete![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
