"""
Example demonstrating web search capabilities for debate evidence gathering.

This example shows both single and multiple argument searches using the
web_search module.
"""

import asyncio
import time

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from llm_debate_assistant.tools.web_search import (
    search_for_evidence,
    search_multiple_arguments,
    search_multiple_arguments_threaded,
)

console = Console()


def display_single_search_example():
    """Demonstrate single argument search with rich formatting."""
    console.print("\n")
    console.print(
        Panel.fit(
            "🔍 Single Argument Search Example",
            style="bold cyan",
        )
    )

    argument = "死刑具有强大的威慑作用，可以有效减少严重犯罪的发生。"
    warrant = (
        "通过执行死刑，潜在的犯罪分子会因为害怕死刑而不敢实施严重犯罪，"
        "从而保护社会安全。"
    )
    evidence_needed = [
        "统计数据：死刑执行前后严重犯罪率的变化",
        "案例研究：具体国家或地区因死刑威慑而犯罪率下降的实例",
        "专家观点：犯罪学家或法律专家对死刑威慑作用的分析",
    ]
    topic = "死刑是否应该被废除"
    side = "正方"

    # Display search parameters
    console.print("\n[bold]Search Parameters:[/bold]")
    console.print(f"  📋 Topic: [cyan]{topic}[/cyan]")
    console.print(f"  ⚖️  Side: [cyan]{side}[/cyan]")
    console.print(f"  💡 Argument: [yellow]{argument}[/yellow]")
    console.print(f"  🔗 Warrant: [yellow]{warrant}[/yellow]")

    # Search with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching for evidence...", total=None)
        start_time = time.perf_counter()
        results = search_for_evidence(
            argument, warrant, evidence_needed, topic, side, model="gemini-2.5-pro"
        )
        elapsed_time = time.perf_counter() - start_time
        progress.update(task, completed=True)

    console.print(f"\n[dim]⏱️  Search completed in {elapsed_time:.2f} seconds[/dim]")

    # Display search results in a table
    if results["search_results"]:
        console.print("\n[bold green]✅ Search Results:[/bold green]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", style="cyan")
        table.add_column("URL", style="blue", overflow="fold")

        for i, res in enumerate(results["search_results"], 1):
            table.add_row(str(i), res["title"], res["uri"])

        console.print(table)
    else:
        console.print("[yellow]No search results found[/yellow]")

    # Display generated response
    if results["text"]:
        console.print("\n[bold green]📝 Generated Analysis:[/bold green]")
        console.print(Panel(results["text"], border_style="green", padding=(1, 2)))
    else:
        console.print("[yellow]No response text generated[/yellow]")


async def display_multiple_search_example(use_threaded: bool = False):
    """Demonstrate multiple argument searches with rich formatting.

    Args:
        use_threaded: If True, uses thread-based async (more reliable for
            concurrent requests)
    """
    console.print("\n\n")

    method_name = "Thread-Based" if use_threaded else "Native Async"
    console.print(
        Panel.fit(
            f"🔍 Multiple Arguments Concurrent Search ({method_name})",
            style="bold magenta",
        )
    )

    topic = "死刑是否应该被废除"
    side = "正方"

    arguments = [
        (
            "死刑具有强大的威慑作用",
            "潜在犯罪分子会因害怕死刑而不敢实施严重犯罪",
            ["统计数据", "案例研究"],
        ),
        (
            "死刑是不可逆的惩罚",
            "一旦执行死刑，如果发现冤案将无法挽回",
            ["冤案统计", "国际人权组织报告"],
        ),
        (
            "死刑成本高于终身监禁",
            "死刑案件的法律程序和关押成本实际上更高",
            ["成本分析报告", "各国司法统计"],
        ),
    ]

    console.print(
        f"\n[bold]Searching for evidence for {len(arguments)} arguments...[/bold]"
    )
    console.print(f"  📋 Topic: [cyan]{topic}[/cyan]")
    console.print(f"  ⚖️  Side: [cyan]{side}[/cyan]\n")

    # Search with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Running {len(arguments)} concurrent searches ({method_name})...",
            total=None,
        )
        start_time = time.perf_counter()
        if use_threaded:
            multiple_results = await search_multiple_arguments_threaded(
                arguments, topic, side
            )
        else:
            multiple_results = await search_multiple_arguments(arguments, topic, side)
        elapsed_time = time.perf_counter() - start_time
        progress.update(task, completed=True)

    console.print(
        f"\n[bold]⏱️  Total time: {elapsed_time:.2f}s | "
        f"Average per argument: {elapsed_time/len(arguments):.2f}s[/bold]"
    )

    # Display results for each argument
    for i, result in enumerate(multiple_results, 1):
        console.print(f"\n[bold cyan]─── Argument {i} ───[/bold cyan]")
        console.print(f"[yellow]Argument:[/yellow] {result.argument}")
        console.print(f"[yellow]Warrant:[/yellow] {result.warrant}")

        if result.error:
            console.print(f"[bold red]❌ Error:[/bold red] {result.error}")
        else:
            num_results = len(result.results.get("search_results", []))
            text_length = len(result.results.get("text", ""))

            console.print(f"[green]✅ Found {num_results} search results[/green]")
            console.print(
                f"[green]📝 Response length: {text_length} characters[/green]"
            )

            # Show search results in a compact table
            if result.results.get("search_results"):
                table = Table(
                    show_header=True,
                    header_style="bold magenta",
                    box=None,
                    padding=(0, 1),
                )
                table.add_column("#", style="dim", width=3)
                table.add_column("Source", style="cyan", overflow="fold")

                for j, res in enumerate(result.results["search_results"][:3], 1):
                    table.add_row(str(j), f"{res['title']}")

                console.print(table)

                if num_results > 3:
                    console.print(
                        f"  [dim]... and {num_results - 3} more results[/dim]"
                    )


async def main():
    """Run all examples."""
    console.print(
        Panel.fit(
            "🎯 Web Search for Debate Evidence - Examples",
            style="bold white on blue",
            padding=(1, 10),
        )
    )

    try:
        # Run single search example
        display_single_search_example()

        # Run multiple search example with native async
        console.print(
            "\n[bold yellow]Testing native async implementation...[/bold yellow]"
        )
        await display_multiple_search_example(use_threaded=False)

        # Run multiple search example with thread-based async
        console.print(
            "\n[bold yellow]Testing thread-based implementation...[/bold yellow]"
        )
        await display_multiple_search_example(use_threaded=True)

        console.print("\n")
        console.print(
            Panel.fit(
                "✨ All examples completed successfully!",
                style="bold green",
            )
        )

    except Exception as e:
        console.print(f"\n[bold red]❌ Error running examples:[/bold red] {str(e)}")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
