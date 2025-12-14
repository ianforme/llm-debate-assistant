"""
Example: Constructive Speech Generation

This example demonstrates how to use the constructive_speech segment to generate
a constructive speech (case construction) based on completed topic_research.

Prerequisites:
- topic_research must be completed first
- Session must exist with topic+side metadata
"""

import asyncio
import logging
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler

from llm_debate_assistant.core.initialization import init_app
from llm_debate_assistant.agents.deep_preparation.storage import (
    create_filesystem,
    find_session_by_topic,
)
from llm_debate_assistant.agents.deep_preparation.segments.constructive_speech import (
    create_constructive_graph,
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


def count_visible_chars(text: str) -> int:
    """Count visible characters (excluding spaces, newlines, tabs)."""
    return len("".join(c for c in text if not c.isspace()))


async def main():
    """Run constructive speech generation example."""

    # Initialize application (prompt manager, etc.)
    console.print("\n[bold blue]Initializing application...[/bold blue]")
    try:
        await init_app()
    except Exception as e:
        console.print(f"[yellow]Warning: Could not initialize Opik: {e}[/yellow]")
        console.print("[yellow]Continuing without Opik integration...[/yellow]")

    # Example topic and side (using cached topic_research)
    topic = "应该强制要求大型科技公司开源其核心算法"
    our_side = "正方"

    console.print(
        Panel.fit(
            f"[bold]Topic:[/bold] {topic}\n[bold]Side:[/bold] {our_side}",
            title="Constructive Speech Generation",
            border_style="blue",
        )
    )

    # Find topic_research session
    console.print("\n[dim]Searching for topic_research session...[/dim]")
    research_session = find_session_by_topic(
        topic=topic,
        side=our_side,
        segment="topic_research",
    )

    if not research_session:
        console.print(
            "[red]✗ No topic_research found for this topic+side![/red]\n"
            "[yellow]Please run topic_research first:[/yellow]\n"
            "  python examples/topic_research_example.py"
        )
        return

    relative_research_path = Path(research_session).relative_to(Path.cwd())
    console.print(f"[green]✓ Found research session: {relative_research_path}[/green]")

    # Create or find constructive_speech session
    console.print("\n[dim]Searching for existing constructive_speech session...[/dim]")
    filesystem, session_path, cache_found = create_filesystem(
        filesystem_type="disk",
        topic=topic,
        side=our_side,
        segment="constructive_speech",
        use_cache=True,
    )

    if cache_found:
        console.print("[green]✓ Found existing constructive_speech session[/green]")
        # Load existing draft if available
        try:
            draft_result = filesystem.read("/constructive_speech/final_speech.md")
            if draft_result.get("success"):
                draft_content = draft_result["content"]
                visible_chars = count_visible_chars(draft_content)
                total_chars = len(draft_content)
                console.print(
                    f"\n[cyan]Existing final speech found "
                    f"({visible_chars} visible chars, {total_chars} total)[/cyan]"
                )
                preview = draft_content[:200] + "..." if len(draft_content) > 200 else draft_content
                console.print(Panel(preview, title="Preview", border_style="cyan"))

                user_input = input("\nRegenerate constructive speech? (y/N): ")
                if user_input.lower() != "y":
                    console.print("[yellow]Using existing speech. Exiting.[/yellow]")
                    return
            else:
                console.print("[blue]No existing draft, will generate new one[/blue]")
        except Exception as e:
            console.print(f"[dim]Could not load existing draft: {e}[/dim]")
            console.print("[blue]Will generate new one[/blue]")
    else:
        console.print("[blue]⚡ Creating new constructive_speech session[/blue]")

    if session_path:
        relative_path = Path(session_path).relative_to(Path.cwd())
        console.print(f"[dim]Session path: {relative_path}[/dim]")

    # Create constructive speech workflow
    console.print("\n[bold blue]Creating constructive speech workflow...[/bold blue]")
    graph = create_constructive_graph()
    app = graph.compile()

    # Initial state
    initial_state = {
        "topic": topic,
        "side": our_side,
        "iteration_count": 0,
        "time_limit": "4 minutes",
        "word_count_limit": 1200,
    }

    # Configuration with filesystem
    config = {
        "configurable": {
            "filesystem": filesystem,
        }
    }

    # Run the workflow
    console.print("\n[bold green]Running constructive speech generation...[/bold green]")
    console.print("[dim](This will take 2-3 minutes for deep evidence search)[/dim]\n")

    start_time = time.time()

    try:
        # Stream progress
        step_emojis = {
            "load_research": "📚",
            "generate_strategy": "🎯",
            "deep_evidence": "🔍",
            "draft_constructive_speech": "✍️",
            "critique_constructive_speech": "📊",
            "finalize_constructive_speech": "🏁",
        }

        final_state = None
        async for event in app.astream(initial_state, config=config):
            for node_name, node_output in event.items():
                emoji = step_emojis.get(node_name, "⚙️")
                console.print(f"{emoji} [bold]{node_name}[/bold] completed")
                # Update final_state with each event
                if isinstance(node_output, dict):
                    if final_state is None:
                        final_state = initial_state.copy()
                    final_state.update(node_output)

        execution_time = time.time() - start_time

        # Display results
        console.print("\n" + "=" * 80)
        console.print(
            Panel.fit(
                f"[bold green]✓ Constructive Speech Generated in "
                f"{execution_time:.1f}s[/bold green]",
                border_style="green",
            )
        )

        # Strategy
        strategy = final_state.get("constructive_strategy")
        if strategy:
            console.print("\n[bold cyan]Strategic Blueprint:[/bold cyan]")
            console.print(f"  Key Terms: {', '.join(t.term for t in strategy.selected_key_terms)}")
            console.print(f"  Arguments: {len(strategy.selected_arguments)}")
            for i, arg in enumerate(sorted(strategy.selected_arguments, key=lambda x: x.order)):
                console.print(f"    {i+1}. {arg.role}: {arg.claim[:60]}...")
            console.print(f"  Speech Tone: {strategy.speech_tone}")
            console.print(f"  Value Premise: {strategy.value_premise[:80]}...")

        # Evidence
        evidence = final_state.get("deep_evidence")
        if evidence:
            total_sources = sum(len(ev.sources) for ev in evidence)
            total_stats = sum(len(ev.statistics) for ev in evidence)
            total_quotes = sum(len(ev.best_quotes) for ev in evidence)
            console.print("\n[bold cyan]Evidence Gathered:[/bold cyan]")
            console.print(f"  Total Sources: {total_sources}")
            console.print(f"  Statistics: {total_stats}")
            console.print(f"  Quotes: {total_quotes}")
            console.print(f"  Arguments Covered: {len(evidence)}")

        # Critique
        critique = final_state.get("critique")
        if critique:
            result_color = "green" if critique.decision == "pass" else "yellow"
            console.print("\n[bold cyan]Final Critique:[/bold cyan]")
            console.print(
                f"  Decision: [{result_color}]{critique.decision.upper()}[/{result_color}]"
            )
            console.print(f"  Score: {critique.score}/10")
            console.print(f"  Iterations: {final_state['iteration_count']}")
            if critique.decision != "pass" and critique.critical_issues:
                console.print("  Outstanding Issues:")
                for issue in critique.critical_issues[:3]:
                    console.print(f"    - {issue}")

        # Draft
        draft = final_state.get("final_speech_content") or final_state.get("draft_content")
        if draft:
            visible_chars = count_visible_chars(draft)
            total_chars = len(draft)
            console.print(
                f"\n[bold cyan]Constructive Speech "
                f"({visible_chars} visible chars, {total_chars} total):[/bold cyan]"
            )
            console.print(Panel(draft, title="Final Speech", border_style="green"))

        # Metadata
        metadata = final_state.get("final_metadata")
        if metadata:
            console.print("\n[bold cyan]Generation Metadata:[/bold cyan]")
            console.print(f"  Status: {metadata.get('status')}")
            console.print(f"  Final Score: {metadata.get('final_score')}/10")
            console.print(f"  Total Iterations: {metadata.get('iterations')}")

        # Session info
        relative_path = Path(session_path).relative_to(Path.cwd())
        console.print(f"\n[dim]All files saved to: {relative_path}[/dim]")
        console.print("[dim]Files:[/dim]")
        console.print("[dim]  - /constructive_speech/strategy.json[/dim]")
        console.print("[dim]  - /constructive_speech/deep_evidence.json[/dim]")
        console.print("[dim]  - /constructive_speech/final_speech.md[/dim]")
        console.print("[dim]  - /constructive_speech/meta.json[/dim]")

    except FileNotFoundError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        console.print("[yellow]Make sure topic_research was completed first![/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during generation: {e}[/red]")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
