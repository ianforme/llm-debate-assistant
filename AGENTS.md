# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Non-Obvious Project Rules
- **LLM Instantiation**: ALWAYS use `llm_debate_assistant.services.llm.get_llm()` instead of instantiating `ChatOpenAI` or `ChatGoogleGenerativeAI` directly. This handles API keys and model-specific configs (like thinking budgets).
- **Gemini Quirk**: When using Gemini models, DO NOT use `.with_structured_output()`. Use the LCEL pattern `prompt | llm | parser` instead to avoid serialization errors.
- **Prompt Management**: Prompts are fetched from Opik via `llm_debate_assistant.services.prompt_manager`. Use `pm.get("PROMPT_NAME")` with fallbacks.
- **Filesystem Abstraction**: Do not use `open()` or `os.path` directly for session artifacts. Use `llm_debate_assistant.services.filesystem_protocol.Filesystem` protocol (disk or virtual).
- **Observability**: All LLM calls are tracked via Opik. Ensure `track_openai` is used for client-level tracking if creating a new client.
- **Testing**: `pytest` works, but extensive type-checking exclusions apply (check `pyproject.toml`).
- **Development Focus**: Active development is in `src/llm_debate_assistant/agents/deep_preparation/`. `reflection_pattern/` and `livekit_voice/` are deprecated.
- **Linting**: Ruff is used with line length 120. `E501` is ignored in prompts and some agent files.
