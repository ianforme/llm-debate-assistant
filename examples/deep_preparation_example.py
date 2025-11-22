"""
Example usage of the Agent-Driven Deep Preparation workflow.

This example demonstrates:
1. Agent autonomously deciding which tools to call and when
2. Planning with todos before execution
3. Filesystem-based context management
4. Message history showing agent's reasoning
"""

import asyncio
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from llm_debate_assistant.agents.deep_preparation import (
    create_deep_prep_workflow,
    create_initial_state,
    get_token_usage,
    reset_token_usage,
)
from llm_debate_assistant.agents.deep_preparation.storage import (
    save_final_state_to_filesystem,
)

console = Console()


async def run_agent_driven_example(
    filesystem_type: Literal["disk", "virtual"] = "disk",
    title: str = "Agent-Driven Deep Preparation",
    verbose: bool = False,
):
    """Run agent-driven debate preparation workflow.

    The agent autonomously decides:
    - When to create todos for planning
    - When to create outline
    - When to search for evidence
    - When to draft the statement
    - When to evaluate and improve
    - When it's done

    Args:
        filesystem_type: Either "disk" (persistent) or "virtual" (in-memory)
        title: Title for the example output
        verbose: Enable verbose logging (message stats, token estimates)
    """

    console.print(f"\n[bold cyan]🤖 {title}[/bold cyan]\n")

    # Reset token tracking for this workflow
    reset_token_usage()

    # Create workflow
    console.print(
        f"[yellow]Creating agent-driven workflow "
        f"with {filesystem_type} filesystem...[/yellow]"
    )
    workflow = create_deep_prep_workflow(filesystem_type=filesystem_type)

    # Create initial state using helper
    initial_state = create_initial_state(
        topic="应该强制要求大型科技公司开源其核心算法",
        side="正方",
        user_message=(
            "请帮我准备这个辩题的开篇立论。"
            "要求：破题并创建大纲、搜集论据、撰写立论稿、评估并改进。"
        ),
        max_iterations=3,  # Max times to retry improve→evaluate loop
        verbose=verbose,  # Enable/disable message stats logging
    )

    console.print(f"[green]✓[/green] Topic: {initial_state['topic']}")
    console.print(f"[green]✓[/green] Side: {initial_state['side']}")
    console.print(f"[green]✓[/green] Max iterations: {initial_state['max_iterations']}")
    console.print(
        f"[green]✓[/green] Verbose logging: {initial_state.get('verbose', False)}\n"
    )

    # Run workflow
    console.print("[yellow]🚀 Starting agent workflow...[/yellow]")
    console.print(
        "[dim]The agent will autonomously decide which tools "
        "to call and when...[/dim]\n"
    )

    try:
        # Run the workflow and get final state
        # Set recursion limit to allow the agent to complete all steps
        # Each agent→tools loop counts as one recursion
        # We need: plan, outline, evidence, draft, evaluate, possibly improve
        # That's roughly 6-10 iterations, so 100 is safe
        final_state = await workflow.ainvoke(initial_state, {"recursion_limit": 100})
        console.print("\n[green]✓ Workflow completed successfully![/green]\n")

        # Save complete final state for reference
        save_final_state_to_filesystem(final_state)
        console.print("[green]✓ Final state saved to /final_state.json[/green]\n")

        # Display token usage summary
        token_stats = get_token_usage()
        if token_stats["call_count"] > 0:
            console.print(
                "[bold cyan]📊 Token Usage Summary (Workflow Total)[/bold cyan]\n"
            )
            input_tokens = token_stats["total_input_tokens"]
            output_tokens = token_stats["total_output_tokens"]
            console.print(
                f"  Input tokens (prompts sent to LLM):    "
                f"[cyan]{input_tokens:,}[/cyan]"
            )
            console.print(
                f"  Output tokens (responses from LLM):    "
                f"[cyan]{output_tokens:,}[/cyan]"
            )
            total = (
                token_stats["total_input_tokens"] + token_stats["total_output_tokens"]
            )
            console.print(
                f"  Total tokens:                          [cyan]{total:,}[/cyan]"
            )
            call_count = token_stats["call_count"]
            console.print(
                f"  Number of LLM calls:                   "
                f"[cyan]{call_count}[/cyan]\n"
            )
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        import traceback

        console.print(f"[red]{traceback.format_exc()}[/red]")
        return

    # Display agent's decision-making process
    console.print("[bold cyan]💬 Agent's Message History[/bold cyan]\n")

    messages = final_state.get("messages", [])
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__

        if msg_type == "SystemMessage":
            # Don't show system prompt (too long)
            if i == 0:
                console.print(f"[dim]{i+1}. [SystemMessage] Initial instructions[/dim]")
        elif msg_type == "HumanMessage":
            console.print(f"{i+1}. [bold cyan][User][/bold cyan] {msg.content}")
        elif msg_type == "AIMessage":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_names = [tc["name"] for tc in msg.tool_calls]
                tools_str = ", ".join(tool_names)
                console.print(
                    f"{i+1}. [bold blue][Agent][/bold blue] "
                    f"Calling tools: {tools_str}"
                )
            else:
                # Final response - use .text for Gemini 3 Pro compatibility
                msg_text = msg.text if hasattr(msg, "text") else str(msg.content)
                console.print(f"{i+1}. [bold blue][Agent][/bold blue] {msg_text}")
        elif msg_type == "ToolMessage":
            console.print(
                f"{i+1}. [bold green][Tool: {msg.name}][/bold green] {msg.content}"
            )

    console.print()

    # Display todos progress
    if final_state.get("todos"):
        console.print("[bold cyan]📋 Task Planning (Todos)[/bold cyan]\n")

        todos = final_state["todos"]
        table = Table(title="Agent's Task List")
        table.add_column("Status", style="cyan", width=12)
        table.add_column("Task", style="white")

        for todo in todos:
            status_emoji = {
                "pending": "⏸️  Pending",
                "in_progress": "🔄 In Progress",
                "completed": "✅ Completed",
            }
            status_display = status_emoji.get(todo["status"], todo["status"])
            table.add_row(status_display, todo["content"])

        console.print(table)
        console.print()

    # Display results
    console.print("[bold cyan]📊 Results[/bold cyan]\n")

    # Show iteration count
    console.print(
        f"Total iterations: [cyan]{final_state.get('iteration_count', 1)}[/cyan]"
    )

    # Show evaluation result
    if final_state.get("evaluation"):
        evaluation = final_state["evaluation"]
        eval_result = evaluation.get("evaluation_result", "unknown")
        console.print(f"Final evaluation: [cyan]{eval_result}[/cyan]")

    # Show filesystem location
    if final_state.get("filesystem_path"):
        console.print(
            f"Files saved to: [cyan]{final_state['filesystem_path']}[/cyan]\n"
        )

    # Get filesystem from state
    filesystem = final_state.get("filesystem")
    if not filesystem:
        console.print("[red]No filesystem found in state[/red]")
        return

    # Display filesystem contents
    console.print("[bold cyan]📁 Filesystem Contents[/bold cyan]\n")

    # List root directory
    root_result = filesystem.ls("/")
    if root_result["success"]:
        table = Table(title="Root Directory")
        table.add_column("File/Directory", style="cyan")

        for item in root_result["items"]:
            table.add_row(item)

        console.print(table)
        console.print()

    # Display outline
    console.print("[bold cyan]📋 Outline[/bold cyan]\n")
    outline_result = filesystem.read("/outline.md")
    if outline_result["success"]:
        outline_content = outline_result["content"]
        console.print(Panel(Markdown(outline_content), border_style="blue"))
        console.print()

    # Display final draft
    console.print("[bold cyan]📝 Final Draft[/bold cyan]\n")
    draft_result = filesystem.read("/current_draft.md")
    if draft_result["success"]:
        draft_content = draft_result["content"]
        console.print(Panel(Markdown(draft_content), border_style="green"))
        console.print()

    # Display final task list status
    if final_state.get("todos"):
        console.print("[bold cyan]📋 Final Task Status[/bold cyan]\n")

        todos = final_state["todos"]
        table = Table(title="Completed Tasks")
        table.add_column("Status", style="cyan", width=12)
        table.add_column("Task", style="white")

        for todo in todos:
            status_emoji = {
                "pending": "⏸️  Pending",
                "in_progress": "🔄 In Progress",
                "completed": "✅ Completed",
            }
            status_display = status_emoji.get(todo["status"], todo["status"])
            table.add_row(status_display, todo["content"])

        console.print(table)
        console.print()

    # Summary
    console.print("[bold green]✨ Agent-driven workflow completed![/bold green]")

    if filesystem_type == "disk":
        console.print(
            f"\n[dim]All work products are saved to:[/dim]"
            f"\n[cyan]{filesystem.get_session_path()}[/cyan]"
        )
        console.print("[dim]You can inspect files with: ls, cat, VSCode, etc.[/dim]\n")


async def main():
    """Run all examples."""

    console.print("[bold yellow]" + "=" * 80)
    console.print("[bold yellow]Agent-Driven Deep Preparation Examples")
    console.print("[bold yellow]" + "=" * 80 + "\n")

    console.print(
        "[dim]Agent autonomously manages the entire debate preparation process[/dim]\n"
    )
    await run_agent_driven_example(
        filesystem_type="disk",
        title="Agent-Driven Workflow with Disk Storage",
        verbose=True,  # Enable message stats logging
    )

    console.print("\n[bold green]🎉 All examples completed![/bold green]\n")


if __name__ == "__main__":
    asyncio.run(main())
