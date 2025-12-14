"""Simple example - workflow for generating a chinese debate opening statement."""

import asyncio
import json
import logging
import time

from opik.integrations.langchain import OpikTracer
from rich.console import Console
from rich.logging import RichHandler
from rich.syntax import Syntax

from llm_debate_assistant.agents.reflection_pattern import create_debate_workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger(__name__)

console = Console()


async def main():
    """Run debate workflow."""

    logger.info("🚀 Starting debate workflow example")

    console.print("\n[bold]Starting workflow with Gemini Pro...[/bold]\n")
    logger.info("📋 Creating workflow graph...")

    # Create workflow
    workflow = create_debate_workflow()
    opik_tracer = OpikTracer(
        graph=workflow.get_graph(xray=True), project_name="llm-debate-assistant"
    )

    # Save workflow graph as image
    try:
        graph_image = workflow.get_graph(xray=True).draw_mermaid_png()
        with open(
            "./asset/opening_statement_workflow.png",
            "wb",
        ) as f:
            f.write(graph_image)
        logger.info("💾 Workflow graph saved to asset/opening_statement_workflow.png")
    except Exception as e:
        logger.warning(f"⚠️  Could not save workflow graph: {e}")

    logger.info("✅ Workflow graph created")

    # Create initial state (simplified - no model config)
    logger.info("🔧 Initializing state...")
    initial_state = {
        "topic": "远程工作是否应该成为新常态",
        "side": "正方",
        "outline": None,
        "evidence_data": None,
        "draft": None,
        "evaluation": None,
        "iteration_count": 1,  # Start at 1 (first draft is iteration 1)
        "max_iterations": 3,  # Will allow up to 3 iteration count
        "messages": [],
        "next_action": "auto",
    }
    logger.info(f"📌 Topic: {initial_state['topic']}")
    logger.info(f"📌 Side: {initial_state['side']}")
    logger.info("📌 Model: gemini-2.5-pro (hardcoded for now)")

    # Run workflow
    console.print("\n[yellow]Running workflow...[/yellow]\n")
    logger.info("🏃 Starting workflow execution...")

    start_time = time.time()
    result = await workflow.ainvoke(initial_state, config={"callbacks": [opik_tracer]})
    end_time = time.time()

    elapsed_time = end_time - start_time

    logger.info("🎉 Workflow completed successfully!")
    logger.info(
        f"⏱️  Total execution time: {elapsed_time:.2f} seconds " f"({elapsed_time/60:.2f} minutes)"
    )

    # Log complete state for debugging
    logger.info("\n📋 Complete State Dictionary:")
    state_copy = dict(result)
    # Convert messages to string representation for readability
    if "messages" in state_copy:
        state_copy["messages"] = [
            (
                f"{type(msg).__name__}: {msg.content[:100]}..."
                if hasattr(msg, "content")
                else str(msg)
            )
            for msg in state_copy["messages"]
        ]

    state_json = json.dumps(state_copy, ensure_ascii=False, indent=2, default=str)
    syntax = Syntax(state_json, "json", theme="monokai", line_numbers=True)
    console.print(syntax)

    # Show results
    console.print("\n[green]✅ Workflow completed![/green]")
    console.print(
        f"[bold]⏱️  Execution time:[/bold] {elapsed_time:.2f}s " f"({elapsed_time/60:.2f}min)"
    )
    console.print(f"\n[bold]Evaluation:[/bold] {result['evaluation']['evaluation_result']}")
    console.print(f"[bold]Iterations:[/bold] {result['iteration_count']}")
    console.print(f"[bold]Word count:[/bold] {len(result['draft'])} characters")

    # Show outline
    logger.info("\n📊 Outline Summary:")
    console.print("\n[bold]Debate Outline:[/bold]")
    outline = result["outline"]
    console.print(f"  Keywords: {len(outline['keyword_definitions'])} defined")
    console.print(f"  Standard: {outline['comparison_standard']['standard'][:100]}...")
    console.print(f"  Arguments: {len(outline['arguments'])} main points")

    # Show draft preview
    logger.info("\n📝 Draft Summary:")
    console.print("\n[bold]Draft Preview:[/bold]")
    console.print(result["draft"] + "...")

    # Show evidence info
    if result.get("evidence_data"):
        total_sources = sum(len(ev.get("search_results", [])) for ev in result["evidence_data"])
        logger.info(f"\n🔍 Evidence: {total_sources} sources gathered")
        console.print(f"\n[bold]Evidence:[/bold] {total_sources} sources gathered")

    return result


if __name__ == "__main__":
    asyncio.run(main())
