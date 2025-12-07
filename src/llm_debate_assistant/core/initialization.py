import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from opik import Opik

logger = logging.getLogger(__name__)


async def init_prompt_manager(opik_client: Optional["Opik"] = None) -> None:
    """Initialize the PromptManager with Opik prompts.

    This should be called once at application startup to load all prompts
    from the Opik platform into the PromptManager cache. Refer to examples/topic_research_example.py
    for an example of usage, ctrl + F "init_prompt_manager".

    Args:
        opik_client (Optional["Opik"]): Optional Opik client instance. If None, will create one.

    Example:
        >>> import opik
        >>> from llm_debate_assistant.init import init_prompt_manager
        >>>
        >>> # Option 1: Provide your own client
        >>> client = opik.Opik()
        >>> await init_prompt_manager(client)
        >>>
        >>> # Option 2: Let it create the client
        >>> await init_prompt_manager()

    Note:
        Requires COMET_API_KEY environment variable to be set.
    """
    from llm_debate_assistant.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()

    # Create Opik client if not provided
    if opik_client is None:
        try:
            import opik

            opik_client = opik.Opik()
            logger.info("Created Opik client")
        except Exception as e:
            logger.error(f"Failed to create Opik client: {e}")
            logger.info("Make sure COMET_API_KEY is set in your environment")
            raise

    # Initialize PromptManager with Opik
    await pm.init(opik_client)
    logger.info(f"PromptManager initialized with {len(pm.list_prompts())} prompts from Opik")


async def init_app(opik_client: Optional["Opik"] = None) -> None:
    """Initialize the entire application.

    This is a convenience function that initializes all required services.
    Currently initializes:
    - PromptManager with Opik prompts

    Args:
        opik_client (Optional["Opik"]): Optional Opik client instance

    Example:
        >>> from llm_debate_assistant.init import init_app
        >>> await init_app()
    """
    logger.info("Initializing llm-debate-assistant...")

    # Initialize PromptManager
    await init_prompt_manager(opik_client)

    logger.info("Initialization complete")
