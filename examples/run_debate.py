import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_debate_assistant.main import match_preparation, simulate_match

if __name__ == "__main__":
    # This is a placeholder for running the debate simulation.
    # You can replace this with actual command-line argument parsing
    # and execution logic.
    import asyncio

    async def main():
        topic = "The rise of artificial intelligence is a threat to humanity."
        preparation_results = await match_preparation(topic)
        simulation_results = await simulate_match(topic, **preparation_results)
        print(simulation_results)

    asyncio.run(main())
