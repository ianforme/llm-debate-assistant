# -*- coding: utf-8 -*-
"""
Topic research agent.

Technically not an "agent" in the LangChain sense due to deterministic execution,
but a coordinating class that orchestrates multiple operations to perform comprehensive topic research.

It works like a sub-agent within the deep preparation framework, focusing on:
1. Defining key terms strategically
2. Researching both sides' arguments in parallel
3. Performing comparative analysis
4. Providing strategic recommendations
"""

import asyncio
from typing import Literal, Optional

from langchain_core.runnables import RunnableConfig

from llm_debate_assistant.agents.deep_preparation.segments.topic_research.schema import (
    TopicResearchResult,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.operations import (
    define_key_terms,
    research_perspective,
    comparative_analysis,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.operations.helpers import (
    opposite_side,
)
from llm_debate_assistant.agents.deep_preparation.segments.topic_research.storage import (
    save_research_to_filesystem,
    load_research_from_filesystem,
)
from llm_debate_assistant.agents.deep_preparation.schema import Filesystem


class TopicResearchAgent:
    async def research(
        self,
        topic: str,
        our_side: Literal["正方", "反方"],
        config: RunnableConfig,
        filesystem: Optional[Filesystem] = None,
        use_cache: bool = True,
    ) -> TopicResearchResult:
        """Conduct comprehensive topic research.

        Args:
            topic (str): The debate topic to research
            our_side (Literal["正方", "反方"]): Which side we are arguing for
            config (RunnableConfig): Runnable configuration for LLM calls
            filesystem (Optional[Filesystem]): Optional filesystem for caching results
            use_cache (bool): Whether to use cached research if available (default: True)

        Returns:
            TopicResearchResult: Complete research result with strategic insights

        Example:
            >>> from llm_debate_assistant.agents.deep_preparation.storage import (
            ...     create_filesystem
            ... )
            >>> filesystem, _ = create_filesystem("disk")
            >>> agent = TopicResearchAgent()
            >>> result = await agent.research(
            ...     topic="应该强制要求大型科技公司开源其核心算法",
            ...     our_side="正方",
            ...     config=config,
            ...     filesystem=filesystem
            ... )
            >>> print(f"Found {len(result.key_terms)} key terms")
            >>> print(f"Our side has {len(result.our_research.arguments)} arguments")
        """
        # Check cache if filesystem is provided and use_cache is True
        if filesystem and use_cache:
            cached_result = load_research_from_filesystem(filesystem)
            if cached_result:
                # Verify cached result matches current request
                if cached_result.topic == topic and cached_result.our_side == our_side:
                    return cached_result

        # Step 1: Define key terms strategically
        key_terms = await define_key_terms(topic, our_side, config)

        # Step 2: Research both perspectives in parallel
        opponent_side = opposite_side(our_side)

        # TODO: Consider evaluating performance impact of asyncio.gather vs sequential calls
        our_research, opponent_research = await asyncio.gather(
            research_perspective(topic, our_side, config),
            research_perspective(topic, opponent_side, config),
        )

        # Step 3: Comparative analysis
        analysis = await comparative_analysis(our_research, opponent_research, config)

        result = TopicResearchResult(
            topic=topic,
            our_side=our_side,
            key_terms=key_terms,
            our_research=our_research,
            opponent_research=opponent_research,
            analysis=analysis,
        )

        # Save to filesystem if provided
        if filesystem:
            save_research_to_filesystem(result, filesystem)

        return result
