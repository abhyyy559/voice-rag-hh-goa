"""Config names the rag-local-eval-loop reads if present (all optional)."""

# Retrieval latency budget shown in the suite's latency report — matches the
# task spec's 200 ms end-to-end target (our measured p95 is ~2 ms).
LATENCY_BUDGET_MS = 200

GENERATION_MODEL = "fast-extractive"
