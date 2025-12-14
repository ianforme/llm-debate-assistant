"""
Example demonstrating web search capabilities using Google Search Grounding.

This example shows both single query and batch query searches using the
web_search service module.
"""

import asyncio
import time

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from llm_debate_assistant.services.web_search import search_queries, search_single_query

console = Console()


async def display_single_search_example():
    """Demonstrate single query search with rich formatting."""
    console.print("\n")
    console.print(
        Panel.fit(
            "🔍 Single Query Search Example",
            style="bold cyan",
        )
    )

    query = "死刑威慑作用 犯罪率统计数据"

    # Display search parameters
    console.print("\n[bold]Search Query:[/bold]")
    console.print(f"  🔎 {query}")

    # Search with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching...", total=None)
        start_time = time.perf_counter()
        result = await search_single_query(query, model="gemini-2.5-flash")
        elapsed_time = time.perf_counter() - start_time
        progress.update(task, completed=True)

    console.print(f"\n[dim]⏱️  Search completed in {elapsed_time:.2f} seconds[/dim]")

    # Check for errors
    if result.error:
        console.print(f"[bold red]❌ Error:[/bold red] {result.error}")
        return

    # Display search sources in a table
    if result.sources:
        console.print("\n[bold green]✅ Sources Found:[/bold green]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", style="cyan")
        table.add_column("URL", style="blue", overflow="fold")

        for i, source in enumerate(result.sources, 1):
            table.add_row(str(i), source.title, source.uri)

        console.print(table)
    else:
        console.print("[yellow]No sources found[/yellow]")

    # Display summarized content
    if result.content:
        console.print("\n[bold green]📝 AI-Generated Summary:[/bold green]")
        console.print(Panel(result.content, border_style="green", padding=(1, 2)))
    else:
        console.print("[yellow]No content generated[/yellow]")


async def display_batch_search_example():
    """Demonstrate batch query searches with rich formatting."""
    console.print("\n\n")

    console.print(
        Panel.fit(
            "🔍 Batch Query Search Example",
            style="bold magenta",
        )
    )

    # Multiple queries on debate-related topics
    queries = [
        "死刑威慑作用 犯罪率统计",
        "死刑冤案案例 中国",
        "死刑成本 终身监禁成本对比",
        "国际人权组织 死刑立场报告",
    ]

    console.print(f"\n[bold]Searching for {len(queries)} queries...[/bold]\n")

    # Display queries
    for i, query in enumerate(queries, 1):
        console.print(f"  {i}. [cyan]{query}[/cyan]")

    # Search with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Running {len(queries)} concurrent searches...",
            total=None,
        )
        start_time = time.perf_counter()
        results = await search_queries(queries, model="gemini-2.5-flash")
        elapsed_time = time.perf_counter() - start_time
        progress.update(task, completed=True)

    console.print(
        f"\n[bold]⏱️  Total time: {elapsed_time:.2f}s | "
        f"Average per query: {elapsed_time / len(queries):.2f}s[/bold]"
    )

    # Display results for each query
    for i, result in enumerate(results, 1):
        console.print(f"\n[bold cyan]─── Query {i} ───[/bold cyan]")
        console.print(f"[yellow]Query:[/yellow] {result.query}")

        if result.error:
            console.print(f"[bold red]❌ Error:[/bold red] {result.error}")
            continue

        num_sources = len(result.sources)
        content_length = len(result.content)

        console.print(f"[green]✅ Found {num_sources} sources[/green]")
        console.print(f"[green]📝 Content length: {content_length} characters[/green]")

        # Show sources in a compact table
        if result.sources:
            table = Table(
                show_header=True,
                header_style="bold magenta",
                box=None,
                padding=(0, 1),
            )
            table.add_column("#", style="dim", width=3)
            table.add_column("Source", style="cyan", overflow="fold")

            # Show first 3 sources
            for j, source in enumerate(result.sources[:3], 1):
                table.add_row(str(j), f"{source.title}")

            console.print(table)

            if num_sources > 3:
                console.print(f"  [dim]... and {num_sources - 3} more sources[/dim]")

        # Show a snippet of the content
        if result.content:
            snippet = result.content[:200] + "..." if len(result.content) > 200 else result.content
            console.print("\n[dim]Content preview:[/dim]")
            console.print(f"  [italic]{snippet}[/italic]")


async def main():
    """Run all examples."""
    console.print(
        Panel.fit(
            "🎯 Web Search Service - Examples",
            style="bold white on blue",
            padding=(1, 10),
        )
    )

    try:
        # Run single search example
        await display_single_search_example()

        # Run batch search example
        await display_batch_search_example()

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
