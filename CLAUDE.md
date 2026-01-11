# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an agentic, LLM-based debate assistant designed to function as an autonomous partner in preparing for Chinese competitive debates (辩论). The system uses LangGraph workflows, ReAct-style orchestration, and specialized subgraphs to prepare debate materials including topic research, constructive speeches, and more.

### Current Development Focus

**This project is currently focused on delivering the `deep_preparation` agent architecture.** All active development should target the `agents/deep_preparation/` module and its subgraphs.

**Deprecated components** (still present but not actively maintained):
- `agents/reflection_pattern/` - Legacy deterministic workflow (superseded by deep_preparation)
- `agents/livekit_voice/` - Voice interaction agent (not current priority)
- `core/assistant.py`, `core/orchestrator.py` - Older core abstractions
- `frontend/app.py` - Streamlit UI (not actively developed)
- Some example scripts in `examples/` (e.g., `run_debate.py`, `reflection_pattern_workflow_example.py`)

**We are NOT cleaning up deprecated code yet** - it remains in the codebase for reference but should not be extended. When working on new features, ignore deprecated components unless explicitly instructed otherwise.

## Development Commands

### Environment Setup
```bash
# Configure poetry to use in-project virtualenvs
poetry config virtualenvs.in-project true

# Install dependencies
poetry install

# Optional: Install audio features (requires PortAudio on macOS)
brew install portaudio
poetry install --extras audio

# Activate virtual environment
source .venv/bin/activate
```

### Code Quality

**Pre-commit hooks** (runs automatically on commit):
```bash
# Install hooks
pre-commit install

# Run all checks manually
pre-commit run --all-files

# Run checks on staged files only
pre-commit run
```

**Individual tools**:
```bash
# Format code with ruff
poetry run ruff format .

# Lint and auto-fix with ruff
poetry run ruff check --fix .

# Type check with mypy (excludes examples/ and notebook/)
poetry run mypy src/
```

**Note**: If mypy blocks commits with unresolvable errors, use `git commit --no-verify` to bypass hooks temporarily.

### Running the Application

**Run debate generation**:
```bash
# Via script
python -m examples.run_debate

# Or use Jupyter notebook
# Open notebook/scratchpad.ipynb
```

**Run the Streamlit UI**:
```bash
streamlit run frontend/app.py
```

**Example workflows**:
```bash
# Deep preparation orchestrator (ReAct agent)
python examples/deep_prep_example.py

# Topic research only
python examples/topic_research_example.py

# Constructive speech generation
python examples/constructive_speech_example.py

# Legacy reflection pattern
python examples/reflection_pattern_workflow_example.py
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_protocol.py

# Run with verbose output
pytest -v
```

## Architecture

### Core Modules (`src/llm_debate_assistant/`)

**`core/`** - Foundation layer
- `client.py` - Centralized OpenAI/Gemini client instances with Opik tracking
- `assistant.py` - Core assistant logic
- `orchestrator.py` - Orchestrator coordination
- `realtime_assistant.py` - LiveKit voice integration
- `initialization.py` - App initialization with Opik setup

**`services/`** - Reusable service layer
- `llm.py` - LLM factory (`get_llm()`) for Gemini/OpenAI with default models
- `prompt_manager.py` - Centralized prompt management via Opik (see Prompt Management section in README)
- `filesystem_protocol.py` - Filesystem abstraction (Protocol)
- `disk_filesystem.py` - Disk-based filesystem implementation
- `virtual_filesystem.py` - In-memory filesystem for testing
- `web_search.py` - Web search service
- `manage_todo_list.py` - Todo list management for agents

**`config/`** - Configuration
- `config.py` - API keys, runtime config (via dataclasses and `.env`)
- `schemas.py` - Pydantic schemas

**`agents/`** - Agent implementations
- `deep_preparation/` - Main ReAct orchestrator + subgraphs (see below)
- `reflection_pattern/` - Legacy deterministic workflow
- `livekit_voice/` - Voice interaction agent

**`helpers/`** - Utility functions
**`prompts/`** - Prompt templates (long Chinese text, E501 ignored in ruff)

### Deep Preparation Architecture

**Orchestrator Pattern** (`agents/deep_preparation/orchestrator.py`):
- ReAct-style agent that autonomously calls subgraph tools
- Tools: `run_topic_research`, `run_constructive_speech`, `update_todo`
- Modes: `lite` (research only) or `full` (research + speech)
- Uses LangGraph's `ToolNode` and `tools_condition` for the agent loop

**Subgraphs** (`agents/deep_preparation/segments/`):
Each segment is a specialized LangGraph workflow wrapped as a tool:

- **`topic_research/`** - Researches both sides, defines key terms, identifies strategic advantages
  - `graph.py` - LangGraph workflow with parallel research nodes
  - `operations/` - Individual operations (define_terms, core_claims, etc.)
  - Uses web search and structured output generation

- **`constructive_speech/`** - Generates opening statement with iterative refinement
  - Creates outline → searches evidence → drafts → critiques → refines
  - Iterative critique loop until quality threshold met
  - `operations/` - Draft, critique, improve operations

- **`closing/`, `rebuttal/`, `tactical_exchange/`** - Work in progress

**Shared Components** (`agents/deep_preparation/shared/`):
- Common state schemas, utilities, and helper functions
- `storage.py` - Artifact persistence to filesystem
- `tools.py` - Tool definitions wrapping subgraphs
- `tracing.py` - Opik integration for LangGraph workflows

### LangGraph Integration

This codebase heavily uses **LangGraph** for agentic workflows:

- **State management**: TypedDict-based state schemas passed between nodes
- **Conditional routing**: Evaluator nodes decide next action (redo_outline, redo_evidence, etc.)
- **Parallel execution**: Multiple research tasks run concurrently
- **Subgraph composition**: Complex workflows built from smaller subgraphs wrapped as tools

**Key Pattern - Subgraph as Tool**:
```python
@tool
async def run_topic_research(topic: str, side: Literal["正方", "反方"]) -> str:
    graph = create_topic_research_graph()
    app = graph.compile()
    final_state = await app.ainvoke(initial_state, config)
    return "✅ Research completed: [summary]"
```

The orchestrator calls these tools using LangGraph's ReAct pattern with `ToolNode`.

### Observability with Opik

**All LLM calls are automatically logged** via Opik integration:

1. **Client-level tracking** (recommended, set in `core/client.py`):
   ```python
   client = track_openai(client, project_name="llm-debate-assistant")
   ```

2. **LangGraph workflow tracking** (for subgraphs):
   ```python
   from opik.integrations.langchain import OpikTracer
   opik_tracer = OpikTracer(graph=workflow.get_graph(xray=True),
                            project_name="llm-debate-assistant")
   result = await workflow.ainvoke(state, config={"callbacks": [opik_tracer]})
   ```

**What's logged**: latency, token counts, cost estimates, model parameters, errors, tool usage, tags (feature/experiment)

**Opik config** (`~/.opik.config`):
```ini
[opik]
url_override = https://www.comet.com/opik/api/
workspace = your-workspace
api_key = your-api-key
```

### Prompt Management

Prompts are **centralized in Opik's Prompt Library** for version control and hot updates:

```python
from llm_debate_assistant.services.prompt_manager import get_prompt_manager

pm = get_prompt_manager()
await pm.init(client)  # Load prompts from Opik into cache

# Use prompts
prompt = pm.get("KEY_TERMS_PROMPT").format(topic=topic, side=side)
result = await llm.ainvoke(prompt, config)
```

Key prompts: `KEY_TERMS_PROMPT`, `CORE_CLAIMS_PROMPT`, `ARGUMENT_DEVELOPMENT_PROMPT`, `VALUE_ADVOCACY_PROMPT`, `COMPARATIVE_ANALYSIS_PROMPT`, `EVIDENCE_EXTRACTION_PROMPT`

## Important Conventions

### Critical Rules (Read First!)

**These non-obvious rules prevent common errors**:

1. **LLM Instantiation**: ALWAYS use `llm_debate_assistant.services.llm.get_llm()` instead of instantiating `ChatOpenAI` or `ChatGoogleGenerativeAI` directly. This handles API keys and model-specific configs (like thinking budgets).

2. **Gemini Quirk**: When using Gemini models, DO NOT use `.with_structured_output()`. Use the LCEL pattern `prompt | llm | parser` instead to avoid serialization errors.

3. **Prompt Management**: Prompts are fetched from Opik via `llm_debate_assistant.services.prompt_manager`. Use `pm.get("PROMPT_NAME")` with fallbacks.

4. **Filesystem Abstraction**: Do not use `open()` or `os.path` directly for session artifacts. Use `llm_debate_assistant.services.filesystem_protocol.Filesystem` protocol (disk or virtual).

5. **Observability**: All LLM calls are tracked via Opik. Ensure `track_openai` is used for client-level tracking if creating a new client.

6. **Development Focus**: Active development is in `src/llm_debate_assistant/agents/deep_preparation/`. `reflection_pattern/` and `livekit_voice/` are deprecated.

### Type Checking with mypy

**mypy excludes**:
- `examples/` - Example scripts ignored
- `notebook/` - Jupyter notebooks ignored
- Many error codes disabled (see `pyproject.toml`)
- Several modules have `ignore_errors = true` (services, deep_preparation segments)

**When adding new code**, prefer enabling type checking unless working in excluded modules.

### Ruff Configuration

- **Line length**: 120 characters
- **E501 (line too long) ignored** in:
  - `prompts/*.py` (contains long Chinese text)
  - `core/*.py`, `agents/**/*.py` (long absolute imports after refactoring)

### Gemini-Specific Best Practices

**Use LCEL chain pattern** instead of `.with_structured_output()` to prevent serialization errors:

```python
# ✅ Recommended for Gemini
chain = prompt | llm | parser
result = await chain.ainvoke(input)

# ❌ Avoid - may cause errors with Gemini
structured_llm = llm.with_structured_output(Schema)
result = await structured_llm.ainvoke(input)
```

**Thinking models** (Gemini 2.5, Gemini 3):
- Automatically configured with `thinking_budget=-1` (unlimited)
- `include_thoughts=False` to avoid empty responses with tool calling

### LLM Provider Configuration

**Default models** (in `services/llm.py`):
- Gemini: `gemini-3-pro-preview`
- OpenAI: `gpt-5.2-2025-12-11`

**Get an LLM instance**:
```python
from llm_debate_assistant.services.llm import get_llm

# Gemini with temperature 0.3
llm = get_llm(temperature=0.3, provider="gemini")

# OpenAI with custom model
llm = get_llm(temperature=0.7, provider="openai", model="gpt-4")
```

### Environment Variables

Required in `.env` (see `.env.example`):
```bash
OPENAI_API_KEY=your-key
ORG_KEY=your-org  # Optional
PROJECT_KEY=your-project  # Optional
GOOGLE_API_KEY=your-gemini-key
```

### Filesystem Abstraction

The codebase uses a **filesystem protocol** for flexibility:

- `filesystem_protocol.py` - Protocol (interface)
- `disk_filesystem.py` - Saves to disk (default)
- `virtual_filesystem.py` - In-memory for testing

**Usage in agents**:
```python
from llm_debate_assistant.agents.deep_preparation import create_filesystem

filesystem, session_path, cache_found = create_filesystem(
    filesystem_type="disk",  # or "virtual"
    topic=topic,
    side=side,
    segment="orchestrator",
    use_cache=True
)
```

## Common Workflows

### Adding a New Debate Segment

1. Create new directory in `src/llm_debate_assistant/agents/deep_preparation/segments/your_segment/`
2. Implement `graph.py` with LangGraph workflow
3. Create operations in `operations/` subdirectory
4. Define prompts in Opik Prompt Library
5. Wrap subgraph as tool in `agents/deep_preparation/tools.py`
6. Add tool to orchestrator in `orchestrator.py`
7. Create example in `examples/your_segment_example.py`

### Debugging LangGraph Workflows

**Enable LangGraph visualization**:
```python
graph = create_your_graph()
print(graph.get_graph().draw_mermaid())  # Mermaid diagram
```

**Stream events for debugging**:
```python
async for event in app.astream(initial_state):
    print(event)  # Inspect state changes per node
```

**Check Opik traces**: All LLM calls and workflow steps are logged to Opik dashboard for inspection.

### Running Single Tests

```bash
# Run specific test class
pytest tests/test_protocol.py::TestFilesystemProtocol

# Run specific test method
pytest tests/test_protocol.py::TestFilesystemProtocol::test_read_file
```

## Code Style Notes

- **AI coding tools welcome** - Just review what you commit, keep it simple, avoid over-engineering
- **No AI slop** - Avoid unnecessary abstractions, bloated code, excessive error handling
- **Type hints required** - Use type annotations throughout
- **Follow existing patterns** - Study similar code before adding new features
- **Async by default** - Most LLM operations are async (`ainvoke`, `astream`)
