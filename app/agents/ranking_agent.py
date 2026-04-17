from __future__ import annotations

from crewai import Agent

from app.tools.llm_tool import RankingTool


def build_ranking_agent() -> Agent:
    return Agent(
        role="Ranking",
        goal='Rank products using the provided tool and return strict JSON.',
        backstory=(
            "You are the primary reasoning agent. Given a user query and retrieval context, "
            "you rank the most relevant products and output STRICT JSON only."
        ),
        allow_delegation=False,
        tools=[RankingTool()],
        max_iter=1,
        verbose=False,
    )

