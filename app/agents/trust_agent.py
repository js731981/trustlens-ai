from __future__ import annotations

from crewai import Agent

from app.tools.scoring_tool import TrustTool


def build_trust_agent() -> Agent:
    return Agent(
        role="Trust",
        goal="Compute trust score using provided tools; output JSON only.",
        backstory="You are a risk and reliability analyst. You score trust and GEO using evidence and return strictly structured JSON.",
        allow_delegation=False,
        tools=[TrustTool()],
        llm=None,
        max_iter=1,
        verbose=False,
    )

