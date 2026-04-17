from __future__ import annotations

from crewai import Task


def build_tasks(
    *,
    retrieval_agent,
    ranking_agent,
    trust_agent,
    analytics_agent,
    explanation_agent,
) -> list[Task]:
    retrieval_task = Task(
        description=(
            "Call the retrieval tool for `{query}`. Return ONLY JSON with keys: "
            "`query`, `intent`, `catalog_names`, `retrieved_documents`, `catalog_source`."
        ),
        agent=retrieval_agent,
        expected_output="A single JSON object (no markdown) with retrieval context.",
    )

    ranking_task = Task(
        description=(
            "Use the prior retrieval JSON context for `{query}` to run the ranking tool.\n"
            "Pass `query={query}`, `provider={provider}`, and `retrieval=<the prior task JSON>`.\n"
            "Return STRICT JSON ONLY with this exact shape (no markdown, no extra keys):\n"
            '{ "ranked_products": [ { "name": "...", "rank": 1 }, ... ] }\n'
        ),
        agent=ranking_agent,
        expected_output='A single JSON object (no markdown) with key "ranked_products".',
        context=[retrieval_task],
    )

    trust_task = Task(
        description=(
            "Use ranking output to compute trust for `{query}`.\n"
            "Pass `query={query}` and `ranking=<the prior task JSON>` to the trust tool.\n"
            "Return ONLY JSON with keys: `geo`, `accuracy`, `trust_score`."
        ),
        agent=trust_agent,
        expected_output="A single JSON object (no markdown) with trust and geo outputs.",
        context=[ranking_task],
    )

    analytics_task = Task(
        description=(
            "Persist run for `{query}`.\n"
            "Use the prior trust output and the provider `{provider}`.\n"
            "Pass `query={query}`, `provider={provider}`, `trust_score=<prior.trust_score>`, `geo=<prior.geo>` to the analytics tool.\n"
            "Return ONLY JSON with key: `saved`."
        ),
        agent=analytics_agent,
        expected_output="A single JSON object (no markdown) acknowledging persistence.",
        context=[trust_task],
    )

    explanation_task = Task(
        description=(
            "Generate a user-facing explanation for `{query}`.\n"
            "Use prior context (ranking/trust/analytics outputs) and return ONLY JSON with keys: `summary`, `insights`.\n"
            "Do not recompute ranking; depend on earlier task outputs."
        ),
        agent=explanation_agent,
        expected_output="A single JSON object (no markdown) with keys: `summary`, `insights`.",
        context=[analytics_task],
    )

    return [retrieval_task, ranking_task, trust_task, analytics_task, explanation_task]

