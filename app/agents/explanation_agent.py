from __future__ import annotations

from crewai import Agent

from app.tools.llm_tool import ExplanationTool


def build_explanation_agent() -> Agent:
    return Agent(
        role="Explanation",
        goal="Summarize ranking; output JSON only.",
        backstory="You are a concise communicator. You explain the ranking outcome clearly and briefly, returning JSON only.",
        allow_delegation=False,
        tools=[ExplanationTool()],
        max_iter=1,
        verbose=False,
    )

