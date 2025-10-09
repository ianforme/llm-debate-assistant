import sys
import os
import asyncio

from llm_debate_assistant.main import match_preparation, simulate_match

if __name__ == "__main__":

    async def main():
        topic = "The rise of artificial intelligence is a threat to humanity."
        preparation_results = await match_preparation(topic)
        simulation_results = await simulate_match(topic, **preparation_results)
        print(simulation_results)

    asyncio.run(main())
