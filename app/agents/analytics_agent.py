from __future__ import annotations

from crewai import Agent

from app.tools.scoring_tool import AnalyticsTool


def build_analytics_agent() -> Agent:
    return Agent(
        role="Analytics",
        goal="Persist the run using provided tools; output JSON only.",
        backstory="You are an analytics operator. You record run metadata and outcomes reliably and return JSON only.",
        allow_delegation=False,
        tools=[AnalyticsTool()],
        llm=None,
        max_iter=1,
        verbose=False,
    )

