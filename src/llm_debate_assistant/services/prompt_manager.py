from typing import TYPE_CHECKING, Dict, Optional
import logging

from llm_debate_assistant.core.prompt_config import OPIK_PROMPT_NAMES

if TYPE_CHECKING:
    from opik import Opik

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompts from managed service or local fallbacks.
    Provides caching and retrieval of prompt templates by name.

    You can either retrieve prompts from a managed service like Opik/LangSmith
    or register them locally from a module until full migration is done.

    regsiter() and register_batch() allow local registration of prompts to the
    in-memory cache.
    """

    def __init__(self):
        """Initialize the prompt manager."""
        self._cache: Dict[str, str] = {}
        self._initialized = False
        self._client: Optional["Opik"] = None  # Opik client instance

    async def init(
        self, client: Optional["Opik"] = None, prompt_names: Optional[list[str]] = None
    ) -> None:
        """Initialize prompts from managed service.

        Loads prompts from Opik (if client provided) or uses local fallbacks.

        Args:
            client (Optional["Opik"]): Optional Opik client instance
            prompt_names (Optional[list[str]]): List of prompt names to fetch from Opik.
                         If None, uses default prompts from core.prompt_config.

        Example:
            >>> import opik
            >>> client = opik.Opik()
            >>> pm = get_prompt_manager()
            >>> await pm.init(client)
        """
        self._client = client

        # Default prompts to fetch from config
        if prompt_names is None:
            prompt_names = OPIK_PROMPT_NAMES

        if client:
            # Load prompts from Opik
            try:
                for name in prompt_names:
                    try:
                        # Get prompt from Opik
                        opik_prompt = client.get_prompt(name=name)
                        # Extract the template string using .prompt attribute
                        self._cache[name] = opik_prompt.prompt
                        logger.debug(f"Loaded prompt from Opik: {name}")
                    except Exception as e:
                        logger.warning(f"Failed to load prompt '{name}' from Opik: {e}")
                        # Continue without this prompt - will use local fallback if available

                logger.info(
                    f"Loaded {len([n for n in prompt_names if n in self._cache])} prompts from Opik"
                )
            except Exception as e:
                logger.error(f"Error initializing Opik prompts: {e}")
                logger.info("Falling back to local prompts")
        else:
            logger.info("No Opik client provided, using local prompts")

        self._initialized = True
        logger.debug(f"Prompt manager initialized with {len(self._cache)} prompts")

    def get(self, name: str, default: str = "") -> str:
        """Get a prompt by name.

        Args:
            name (str): Prompt identifier
            default (str): Default prompt if not found in cache

        Returns:
            str: The prompt template string

        Raises:
            KeyError: If prompt not found and no default provided

        Example:
            >>> pm = get_prompt_manager()
            >>> prompt = pm.get("debate_analysis")
            >>> result = await llm.ainvoke(prompt.format(topic="..."))
        """
        if name not in self._cache and not default:
            raise KeyError(
                f"Prompt '{name}' not found. "
                "Make sure to register it with register() or provide a default."
            )

        return self._cache.get(name, default)

    def register(self, name: str, template: str):
        """Register a prompt template locally.

        This allows modules to register their prompts until we migrate
        to a managed prompt service.

        Args:
            name (str): Unique prompt identifier
            template (str): Prompt template string (may contain {placeholders})

        Example:
            >>> pm = get_prompt_manager()
            >>> pm.register("greeting", "Hello {name}, how can I help?")
        """
        if name in self._cache:
            logger.warning(f"Overwriting existing prompt: {name}")

        self._cache[name] = template
        logger.debug(f"Registered prompt: {name}")

    def register_batch(self, prompts: Dict[str, str]):
        """Register multiple prompts at once.

        Args:
            prompts (Dict[str, str]): Dictionary of {name: template} pairs

        Example:
            >>> pm = get_prompt_manager()
            >>> pm.register_batch({
            ...     "greeting": "Hello {name}",
            ...     "farewell": "Goodbye {name}",
            ... })
        """
        for name, template in prompts.items():
            self.register(name, template)

    def has(self, name: str) -> bool:
        """Check if a prompt is registered.

        Args:
            name (str): Prompt identifier

        Returns:
            bool: True if prompt exists, False otherwise
        """
        return name in self._cache

    def list_prompts(self) -> list[str]:
        """List all registered prompt names.

        Returns:
            List[str]: List of prompt identifiers
        """
        return list(self._cache.keys())

    def clear(self):
        """Clear all cached prompts (useful for testing)."""
        self._cache.clear()
        logger.debug("Cleared all prompts from cache")


# Global singleton instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance.

    This follows the singleton pattern used by get_llm() in services/llm.py.

    Returns:
        PromptManager: The global PromptManager instance

    Example:
        >>> from llm_debate_assistant.services.prompt_manager import get_prompt_manager
        >>> pm = get_prompt_manager()
        >>> pm.register("my_prompt", "...")
        >>> prompt = pm.get("my_prompt")
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


# Convenience function for registering prompts from modules, allowing
# prompt registration at import time if we have yet to migrate to a managed service or
# still want to use local prompts.

# This is useful for modules to register their prompts when they are imported.


def register_prompts(prompts: Dict[str, str]):
    """Convenience function to register prompts at module level.

    This is useful for registering prompts when a module is imported.

    Args:
        prompts (Dict[str, str]): Dictionary of {name: template} pairs

    Example:
        # In your module's __init__.py or prompts.py
        from llm_debate_assistant.services.prompt_manager import register_prompts

        register_prompts({
            "key_terms_analysis": "Analyze the following topic...",
            "perspective_research": "Research arguments for...",
        })
    """
    pm = get_prompt_manager()
    pm.register_batch(prompts)
