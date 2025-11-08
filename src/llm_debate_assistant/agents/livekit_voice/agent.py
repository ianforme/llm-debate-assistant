from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.plugins import google, noise_cancellation

from llm_debate_assistant.prompts.oregon_oxford_prompts import (
    demo_bottomline,
    demo_example,
    demo_statement,
    oregon_interrogated_prompts,
)

# Load environment variables from .env file
# Not following the docs
load_dotenv()


class DebateAssistant(Agent):
    def __init__(
        self,
        topic: str = "台湾是否应该废除移工私人中介制度",
        assistant_side: str = "反方 - 不应该废除",
        debate_baseline: str = demo_bottomline,
        match_history: str = demo_statement,
        assistant_examples: str = demo_example,
    ) -> None:
        # Generate the full system instructions for the debate
        instructions = oregon_interrogated_prompts(
            topic=topic,
            assistant_side=assistant_side,
            debate_baseline=debate_baseline,
            match_history=match_history,
            assistant_examples=assistant_examples,
        )
        super().__init__(instructions=instructions)


async def entrypoint(ctx: agents.JobContext):
    # Create the debate assistant with default or custom parameters
    debate_agent = DebateAssistant()

    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model="gemini-2.5-flash-native-audio-preview-09-2025",
            # _gemini_tools=[types.GoogleSearch()],
        )
    )

    await session.start(
        room=ctx.room,
        agent=debate_agent,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` instead for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Use generate_reply with a simple greeting instruction, not the full prompt
    await session.generate_reply(
        instructions="请用中文接受对方辩友的质询，并进行回应。",
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
