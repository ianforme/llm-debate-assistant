import json
import os

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, JobProcess, RoomInputOptions, inference
from livekit.plugins import google, noise_cancellation, openai, silero

from llm_debate_assistant.agents.livekit_voice.model_config import (
    DebateConfig,
    ModelType,
    get_config,
)
from llm_debate_assistant.prompts.oregon_oxford_prompts import (
    demo_bottomline,
    demo_example,
    demo_statement,
    oregon_interrogated_prompts,
)

# Load environment variables from .env file
load_dotenv()


class DebateAssistant(Agent):
    def __init__(self, debate_config: DebateConfig) -> None:
        """Initialize debate assistant with configuration.

        Args:
            debate_config: Debate-specific configuration including topic, side, etc.
        """
        # Generate the full system instructions for the debate
        instructions = oregon_interrogated_prompts(
            topic=debate_config.topic,
            assistant_side=debate_config.assistant_side,
            debate_baseline=debate_config.debate_baseline,
            match_history=debate_config.match_history,
            assistant_examples=debate_config.assistant_examples,
        )
        super().__init__(instructions=instructions)


def prewarm(proc: JobProcess) -> None:
    """Prewarm function to pre-load heavy models for standard pipeline.

    Args:
        proc (JobProcess): The job process context.
    """
    # Load configuration
    config_name = os.getenv("AGENT_CONFIG")
    config = get_config(config_name)

    # Only load VAD if using standard pipeline
    # Standard pipeline requires STT, LLM, TTS, and VAD components
    if config.model_type == ModelType.STANDARD:
        proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: agents.JobContext) -> None:
    """Main entrypoint of the voice assistant agent.

    Debate parameters can be provided via room metadata when creating the room:
    ```python
    room = await livekit_api.room.create_room(
        livekit.CreateRoomRequest(
            name="debate-room",
            metadata=json.dumps({
                "topic": "Your debate topic",
                "side": "正方 - 赞成",
                "baseline": "...",
                "history": "...",
                "examples": "...",
            })
        )
    )
    ```

    Args:
        ctx (agents.JobContext): job context
    """

    # Load model configuration
    config_name = os.getenv("AGENT_CONFIG")
    config = get_config(config_name)

    # Parse room metadata for debate parameters
    metadata = {}
    if ctx.room.metadata:
        try:
            metadata = json.loads(ctx.room.metadata)
        except json.JSONDecodeError:
            pass

    # Create debate configuration from room metadata or use defaults
    # In production use, these should be provided by the user
    # The config should be filled in on a per-session basis
    # This is just for demonstration purposes
    debate_config = DebateConfig(
        topic=metadata.get("topic", "台湾是否应该废除移工私人中介制度"),
        assistant_side=metadata.get("side", "反方 - 不应该废除"),
        debate_baseline=metadata.get("baseline", demo_bottomline),
        match_history=metadata.get("history", demo_statement),
        assistant_examples=metadata.get("examples", demo_example),
    )

    # Create the debate assistant with user-provided or default parameters
    debate_agent = DebateAssistant(debate_config=debate_config)

    # Create session based on model type
    if config.model_type == ModelType.GOOGLE_REALTIME:
        # Google Realtime Model: end-to-end voice model with built-in VAD
        session = AgentSession(
            llm=google.realtime.RealtimeModel(
                model=config.google_realtime.model,
                # _gemini_tools=[types.GoogleSearch()]
                # if config.google_realtime.enable_google_search else None,
            )
        )
    elif config.model_type == ModelType.OPENAI_REALTIME:
        # OpenAI Realtime Model: end-to-end voice model with built-in VAD
        session = AgentSession(
            llm=openai.realtime.RealtimeModel(
                model=config.openai_realtime.model,
                voice=config.openai_realtime.voice,
            )
        )
    else:
        # Standard Pipeline: separate STT, LLM, TTS, VAD components
        # For weaker models that don't come with built-in VAD
        session = AgentSession(
            stt=inference.STT(
                model=config.standard_pipeline.stt_model,
                language=config.standard_pipeline.stt_language,
            ),
            llm=inference.LLM(
                model=config.standard_pipeline.llm_model,
            ),
            tts=inference.TTS(
                model=config.standard_pipeline.tts_model,
                voice=config.standard_pipeline.tts_voice,
                language=config.standard_pipeline.tts_language,
            ),
            vad=ctx.proc.userdata.get("vad"),  # Use pre-loaded VAD from prewarm
        )

    await session.start(
        room=ctx.room,
        agent=debate_agent,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` instead for best results
            noise_cancellation=(
                noise_cancellation.BVC() if config.enable_noise_cancellation else None
            ),
        ),
    )

    # Use generate_reply with a simple greeting instruction
    await session.generate_reply(
        instructions="请用中文接受对方辩友的质询，并进行回应。在开始之前，请示意对方辩友你已经准备好了，可以开始质询。",
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm)
    )
