# Production-Oriented AI Research Agent

A production-grade, bounded AI Research Agent built using **Python 3.11+**, **LangGraph**, **LangChain Google GenAI (Gemini)**, and **Pydantic**.

## Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure credentials:
   ```bash
   copy .env.example .env
   ```
   Set `GEMINI_API_KEY` and `TAVILY_API_KEY` in `.env`.
   If either credential is missing, the corresponding operation fails
   gracefully and the agent returns a structured low-confidence fallback when
   no verified evidence is available.

3. Run the agent:
   ```bash
   python -m app.main "What are the latest developments in renewable energy?"
   ```

## Testing

Run the unit tests from the generated project directory:

```bash
pytest
```

The evaluation runner performs live Gemini/Tavily calls and writes actual
results with a UTC timestamp to `evaluation/evaluation_results.json`:

```bash
python evaluation/run_evaluation.py
```

Each result records schema validity, grounding, confidence match, per-node
latency, token usage, and the final answer fields. If the default model's
free-tier quota is exhausted, override it:

```bash
GEMINI_MODEL=gemini-2.5-flash-lite python evaluation/run_evaluation.py
```

The latest executed run and its analysis are in
[`evaluation/evaluation_results.json`](evaluation/evaluation_results.json) and
[`evaluation/evaluation_report.md`](evaluation/evaluation_report.md).

## Adaptive research loop

The agent performs a bounded, evidence-driven loop:

1. The planner creates the initial subquestions and searches.
2. The researcher records executed queries and preserves all discovered URLs
   and fetched pages.
3. The evidence processor evaluates relevance and subquestion coverage, then
   records any unanswered subquestions as evidence gaps.
4. If gaps remain, the next research cycle generates new targeted queries for
   those gaps and filters out previously used queries.
5. The graph synthesizes when evidence is sufficient or when
   `MAX_RESEARCH_ITERATIONS` is reached. Tool failures or follow-up planning
   failures use deterministic fallbacks and do not discard collected evidence.
