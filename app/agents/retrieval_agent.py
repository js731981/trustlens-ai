from __future__ import annotations

from crewai import Agent

from app.tools.rag_tool import RetrievalTool


def build_retrieval_agent() -> Agent:
    return Agent(
        role="Retrieval",
        goal="Fetch relevant product context using provided tools.",
        backstory="You are a meticulous research assistant that finds the most relevant product facts from the available knowledge sources.",
        allow_delegation=False,
        tools=[RetrievalTool()],
        llm=None,
        max_iter=1,
        verbose=False,
    )

