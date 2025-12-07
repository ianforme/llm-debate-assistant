"""
Example usage of the Topic Research Graph.

This demonstrates how to use the LangGraph-based workflow to conduct strategic research
on debate topics, researching both sides and providing strategic recommendations.

Prerequisites:
- COMET_API_KEY environment variable must be set
- All prompts must be uploaded to Opik platform
"""

import asyncio
import opik
import time
import logging

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler

from llm_debate_assistant.core import init_prompt_manager, close_async_clients
from llm_debate_assistant.agents.deep_preparation.segments.topic_research import (
    create_topic_research_graph,
    create_initial_state,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.storage import (
    load_research_from_filesystem,
    load_research_progress,
)
from llm_debate_assistant.agents.deep_preparation.storage import (
    create_filesystem,
    save_session_metadata,
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


async def run_topic_research(topic: str, our_side: str):
    """Run topic research and display results.

    Args:
        topic: The debate topic
        our_side: Which side we are arguing for (正方 or 反方)
    """
    console.print("\n[bold cyan]🔍 Topic Research Graph[/bold cyan]\n")
    console.print(f"[yellow]Topic:[/yellow] {topic}")
    console.print(f"[yellow]Our Side:[/yellow] {our_side}")

    # Create filesystem with automatic cache lookup
    console.print("\n[dim]Searching for existing session...[/dim]")
    filesystem, session_path, cache_found = create_filesystem(
        filesystem_type="disk",
        topic=topic,
        side=our_side,
        segment="topic_research",
        use_cache=True,
    )

    if cache_found:
        console.print("[green]✓ Found existing session for this topic+side[/green]")
    else:
        console.print("[blue]⚡ Creating new session[/blue]")
        # Save metadata for future cache lookup
        save_session_metadata(filesystem, topic, our_side, "topic_research")

    if session_path:
        console.print(f"[dim]Session path: {session_path}[/dim]")

    # Check for existing cache/progress
    console.print("\n[bold cyan]📦 Checking for cached/partial research...[/bold cyan]")
    cached_research = load_research_from_filesystem(filesystem)
    partial_progress = load_research_progress(filesystem)

    if cached_research:
        console.print(
            Panel(
                "[green]✅ Complete research found in cache![/green]\n"
                "This will be loaded instead of regenerating.",
                title="Cache Hit",
                border_style="green",
            )
        )
    elif partial_progress:
        stage = partial_progress.get("stage", "unknown")
        stage_names = {
            "terms_complete": "Key Terms Definition",
            "research_complete": "Perspective Research",
            "analysis_complete": "Comparative Analysis",
        }
        stage_display = stage_names.get(stage, stage)
        console.print(
            Panel(
                f"[yellow]⚡ Partial progress found![/yellow]\n"
                f"Stage: [bold]{stage_display}[/bold]\n"
                f"Will resume from checkpoint.",
                title="Partial Cache Hit",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                "[blue]🆕 No cache found[/blue]\n" "Will execute complete workflow from scratch.",
                title="Cache Miss",
                border_style="blue",
            )
        )

    console.print("\n[bold cyan]🚀 Starting Workflow Execution[/bold cyan]\n")

    # Create graph workflow and compile
    graph = create_topic_research_graph()
    app = graph.compile()

    try:
        # Start timing
        start_time = time.time()
        start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        console.print(f"[dim]Start time: {start_time_str}[/dim]")

        # Create initial state and run workflow
        initial_state = create_initial_state(
            topic=topic,
            side=our_side,
            filesystem=filesystem,
            use_cache=True,  # Enable cache/resume
        )

        # Display workflow stages
        console.print("\n[bold]Workflow Stages:[/bold]")
        console.print("  1️⃣  Check Cache")
        console.print("  2️⃣  Define Key Terms")
        console.print("  3️⃣  Research Both Sides")
        console.print("  4️⃣  Comparative Analysis")
        console.print("  5️⃣  Finalize Results\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Executing workflow...", total=None)
            final_state = await app.ainvoke(initial_state, {"configurable": {}})
            progress.update(task, description="[green]✓ Workflow complete")

        result = final_state["research_result"]

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_minutes = elapsed_time / 60

        console.print("[green]✓ Research completed![/green]")
        console.print(
            f"[cyan]⏱️  Total time: {elapsed_time:.2f} seconds "
            f"({elapsed_minutes:.2f} minutes)[/cyan]\n"
        )
        logger.info(f"Topic research completed in {elapsed_time:.2f} seconds")

        # Show incremental saves status
        console.print(
            Panel(
                "[bold]Incremental Saves During Execution:[/bold]\n\n"
                "✓ [green]Key Terms Defined[/green] → "
                "Checkpoint saved to [dim]_progress.json[/dim]\n"
                "✓ [green]Both Sides Researched[/green] → "
                "Checkpoint saved to [dim]_progress.json[/dim]\n"
                "✓ [green]Analysis Completed[/green] → "
                "Checkpoint saved to [dim]_progress.json[/dim]\n"
                "✓ [green]Final Results[/green] → "
                "Saved to [dim]research.json[/dim] + markdown files\n\n"
                "[dim]These checkpoints enable resuming from any stage "
                "if interrupted.[/dim]",
                title="🔄 Resumability",
                border_style="cyan",
            )
        )

        # Display Key Terms
        console.print("[bold cyan]📚 Key Terms (Strategic Definitions)[/bold cyan]\n")
        for term in result.key_terms:
            console.print(f"[bold]{term.term}[/bold]")
            console.print(f"  [dim]中立定义:[/dim] {term.neutral_definition}")
            console.print(f"  [green]我方定义:[/green] {term.our_side_definition}")
            console.print(f"  [yellow]对方定义:[/yellow] {term.opponent_definition}")
            console.print(f"  [blue]战略建议:[/blue] {term.strategic_note}")
            console.print()

        # Display Our Research
        console.print(f"[bold cyan]💪 Our Side ({our_side}) Research[/bold cyan]\n")
        console.print("[bold]Core Claims:[/bold]")
        for i, claim in enumerate(result.our_research.core_claims, 1):
            console.print(f"  {i}. {claim}")
        console.print()

        console.print(f"[bold]Value Framework:[/bold] {result.our_research.value_framework}")
        console.print(
            f"[bold]Comparison Standard:[/bold] {result.our_research.comparison_standard}"
        )
        console.print()

        console.print("[bold]Arguments:[/bold]")
        for i, arg in enumerate(result.our_research.arguments, 1):
            console.print(f"\n  [bold cyan]Argument {i}: {arg.claim}[/bold cyan]")
            # Calculate strength indicator
            strength_dots = 3 if arg.strength == "strong" else 2 if arg.strength == "medium" else 1
            console.print(f"  Strength: [{arg.strength}] {'●' * strength_dots}")
            console.print(f"  Reasoning: {arg.reasoning[:200]}...")
            console.print(f"  Logical Chain: {arg.logical_chain[:150]}...")
            console.print(f"  Evidence: {len(arg.evidence)} pieces")
            if arg.evidence:
                for ev in arg.evidence[:2]:  # Show first 2
                    console.print(f"    - {ev[:100]}...")
        console.print()

        # Display Opponent Research
        opponent_side = "反方" if our_side == "正方" else "正方"
        console.print(f"[bold red]🎯 Opponent ({opponent_side}) Research[/bold red]\n")
        console.print("[bold]Core Claims:[/bold]")
        for i, claim in enumerate(result.opponent_research.core_claims, 1):
            console.print(f"  {i}. {claim}")
        console.print()

        console.print("[bold]Arguments:[/bold]")
        for i, arg in enumerate(result.opponent_research.arguments, 1):
            console.print(f"  {i}. {arg.claim} (Strength: {arg.strength})")
        console.print()

        # Display Comparative Analysis
        console.print("[bold cyan]⚔️  Comparative Analysis[/bold cyan]\n")

        console.print("[bold]Key Clashes:[/bold]")
        for i, clash in enumerate(result.analysis.key_clashes, 1):
            console.print(f"\n  [bold yellow]Clash {i}: {clash.issue}[/bold yellow]")
            console.print(f"  Our Position: {clash.our_position}")
            console.print(f"  Opponent Position: {clash.opponent_position}")
            console.print(f"  Analysis: {clash.analysis[:150]}...")
        console.print()

        console.print("[bold green]Our Advantages:[/bold green]")
        for i, adv in enumerate(result.analysis.our_advantages, 1):
            console.print(f"  {i}. {adv}")
        console.print()

        console.print("[bold yellow]Opponent Vulnerabilities:[/bold yellow]")
        for i, vuln in enumerate(result.analysis.opponent_vulnerabilities, 1):
            console.print(f"  {i}. {vuln}")
        console.print()

        console.print("[bold blue]Strategic Recommendations:[/bold blue]")
        for i, rec in enumerate(result.analysis.strategic_recommendations, 1):
            console.print(f"  {i}. {rec}")
        console.print()

        # Summary Table
        summary_table = Table(title="Research Summary")
        summary_table.add_column("Component", style="cyan")
        summary_table.add_column("Count", style="magenta")

        summary_table.add_row("Key Terms", str(len(result.key_terms)))
        summary_table.add_row("Our Arguments", str(len(result.our_research.arguments)))
        summary_table.add_row("Opponent Arguments", str(len(result.opponent_research.arguments)))
        summary_table.add_row("Key Clashes", str(len(result.analysis.key_clashes)))
        summary_table.add_row("Our Advantages", str(len(result.analysis.our_advantages)))
        summary_table.add_row(
            "Opponent Vulnerabilities",
            str(len(result.analysis.opponent_vulnerabilities)),
        )
        summary_table.add_row(
            "Strategic Recommendations",
            str(len(result.analysis.strategic_recommendations)),
        )

        console.print(summary_table)

        console.print("\n[bold green]✨ Research Complete![/bold green]")
        console.print(
            "[dim]This research can now be used to inform " "opening statement creation.[/dim]"
        )

        # Show filesystem location
        if session_path:
            research_path = f"{session_path}/research/"
            console.print(f"\n[cyan]📁 Research saved to:[/cyan] [dim]{research_path}[/dim]")
            console.print("[dim]Files saved:[/dim]")
            console.print("  - research.json (structured data)")
            console.print("  - _progress.json (incremental checkpoints)")
            console.print("  - research_summary.md")
            console.print("  - key_terms.md")
            console.print("  - our_arguments.md")
            console.print("  - opponent_arguments.md")
            console.print("  - analysis.md")
        console.print()

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        import traceback

        console.print(f"[red]{traceback.format_exc()}[/red]")


async def main():
    """Run examples."""

    console.print("[bold yellow]" + "=" * 80)
    console.print("[bold yellow]Topic Research Graph Examples")
    console.print("[bold yellow]" + "=" * 80 + "\n")

    # Initialize Opik and load prompts
    try:
        console.print("[cyan]Initializing Opik and loading prompts...[/cyan]")
        init_start = time.time()

        client = opik.Opik()
        await init_prompt_manager(client)

        init_elapsed = time.time() - init_start
        console.print(f"[green]✓ Prompts loaded from Opik in {init_elapsed:.2f} seconds[/green]\n")
        logger.info(f"Opik initialization took {init_elapsed:.2f} seconds")
    except Exception as e:
        console.print(f"[red]✗ Failed to initialize Opik: {e}[/red]")
        console.print("[yellow]Make sure COMET_API_KEY is set in your environment[/yellow]")
        return

    # Example 1: Technology topic
    await run_topic_research(topic="应该强制要求大型科技公司开源其核心算法", our_side="正方")

    # Cleanup async resources
    await close_async_clients()


if __name__ == "__main__":
    asyncio.run(main())
