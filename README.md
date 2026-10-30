# Production-Oriented AI Research Agent

A bounded, evidence-driven research agent built with **Python 3.11+**,
**LangGraph**, **LangChain Google GenAI (Gemini)**, **Pydantic**, and
**Tavily**.

The agent accepts a research question, creates an explicit plan, searches and
fetches sources, filters evidence for relevance, decides whether another
research pass is needed, and returns a structured answer with citations,
confidence, assumptions, and limitations.

## Why an agentic approach?

Research questions are not single-shot lookups. The answer depends on which
sources exist, what they contain, and what is still missing after the first
search. A fixed `search -> summarize` pipeline cannot:

- decompose an ambiguous or multi-part question into subquestions;
- choose where to look next when the first results are weak;
- notice that a requested subquestion is unanswered;
- recover when search, fetch, or planning fails; or
- decide when it has enough evidence to stop.

This project uses a small planner-executor loop rather than a large framework.
The loop is bounded to two research iterations, so it adds adaptivity without
allowing runaway cost or latency.

## Architecture

```text
User question
     |
     v
+------------+     structured plan (objective, subquestions, searches, tools)
|  Planner   |-----------------------------------------------------+
+------------+                                                     |
                                                                   v
                                                        +--------------------+
                                                        |     Researcher     |
                                                        | Tavily search      |
                                                        | page fetch/extract |
                                                        +--------------------+
                                                                   |
                                                                   v
                                                        +--------------------+
                                                        |  Evidence Filter   |
                                                        | relevance / dedupe |
                                                        | coverage / gaps    |
                                                        +--------------------+
                                                                   |
                                          sufficient or budget done?
                                               /           \
                                             no             yes
                                             |               |
                                             v               v
                                      +------------+   +--------------+
                                      | Follow-up  |   | Synthesizer  |
                                      | researcher |   | schema +     |
                                      +------------+   | citations    |
                                             |         +--------------+
                                             +-------> final response
```

The graph is assembled in [`app/graph.py`](app/graph.py):

```text
planner -> researcher -> evidence -> (researcher | synthesizer) -> END
```

The conditional edge is bounded by `MAX_RESEARCH_ITERATIONS` (default `2`).
Count-based discovery queries such as "find 5 startups" are forced through one
breadth-refinement pass even if the first pass looks lexically sufficient.

### Nodes

| Node | Responsibility |
|---|---|
| **Planner** | Produces a structured `ResearchPlan`: objective, subquestions, proposed searches, intended tools, evidence needs, completion criteria. On failure, uses a deterministic fallback plan. |
| **Researcher** | Runs selected tools, records every tool call, searches up to three queries, fetches up to five pages, extracts direct URLs from the question, and carries evidence forward across iterations. |
| **Evidence** | Scores relevance, canonicalizes and deduplicates URLs, removes duplicate content within a domain while preserving cross-domain corroboration, computes subquestion coverage, records gaps, and decides sufficiency. |
| **Synthesizer** | Produces a Pydantic-validated `ResearchResponse`, validates citations against retrieved evidence, applies a deterministic confidence calibration, and flags prompt-injection attempts. |

## Tools

### 1. Web search (`app/tools/web_search.py`)

Tavily search with:

- bounded retries and exponential backoff;
- request timeout;
- malformed/missing payload handling;
- per-call latency and status recording;
- search score preservation for relevance ranking.

### 2. Page fetch (`app/tools/fetch_page.py`)

HTML fetch and text extraction with:

- SSRF protection (blocks localhost/private/link-local targets);
- redirect handling and limits;
- request timeout and retries;
- content-size and extracted-text limits;
- graceful handling of HTTP errors, empty pages, and non-HTML responses.

Direct URLs in the user question are also extracted and fetched, so users can
point the agent at specific sources.

## Structured output

All final responses validate against
[`ResearchResponse`](app/schemas.py):

```json
{
  "question": "...",
  "short_answer": "...",
  "key_findings": [
    {"finding": "...", "source_ids": ["S1"]}
  ],
  "sources": [
    {
      "source_id": "S1",
      "title": "...",
      "url": "https://...",
      "relevance_description": "..."
    }
  ],
  "confidence": {
    "level": "High | Medium | Low",
    "justification": "..."
  },
  "assumptions": ["..."],
  "limitations": ["..."],
  "suggested_next_steps": ["..."]
}
```

Structured LLM calls use `with_structured_output` plus retries in
[`app/utils/llm.py`](app/utils/llm.py). If the model returns invalid citations,
the synthesizer removes invalid source IDs and only promotes sources that
surviving findings actually cite.

## Evidence filtering and confidence

Relevance combines lexical overlap (with light stemming and stopword removal)
and the search engine's own score. Sources below the quality threshold are kept
for auditability but do not count as quality evidence.

A research pass is considered sufficient when:

- at least two quality sources exist;
- they span at least two domains; and
- every planned subquestion is covered.

Otherwise, the next iteration receives the uncovered subquestions as gaps and
generates targeted follow-up queries.

For temporal questions containing terms such as "latest", "new", "recent", or
"current", evidence also receives a deterministic recency score from source
publication dates or dates found in the source text. Organization-specific
questions receive an authority preference for known official domains (for
example, `openai.com` for OpenAI). Hardware, infrastructure, product-feature,
and directory results are not allowed to qualify as model-release evidence
solely because they mention the organization. If recent, authoritative,
independently corroborated evidence cannot be established, the answer states
that limitation instead of presenting historical releases as current.

Confidence is calibrated deterministically on top of the model's estimate:

| Condition | Maximum confidence |
|---|---|
| No evidence or no quality evidence | Low |
| Fewer than two quality sources | Low/Medium |
| Ambiguous question, uncovered gaps, one domain, or insufficient evidence | Medium |
| Multiple quality sources, multiple domains, full coverage, unambiguous | High |

This prevents a model from declaring High confidence from one noisy page or on
an intentionally underspecified question.

## Failure handling

| Failure | Behavior |
|---|---|
| Planner LLM timeout/error | Deterministic fallback plan; warning preserved. |
| Follow-up query generation failure | Falls back to gap-derived query or a generic authoritative-evidence query. |
| Search timeout/HTTP error/malformed payload | Retries with backoff; records the error and returns no results for that call. |
| Fetch timeout/HTTP error/private URL | Retries once where appropriate; records per-URL error; other sources continue. |
| Empty search results | Treated as insufficient evidence; the loop refines or reaches the iteration budget. |
| No evidence after the budget | Schema-valid Low-confidence fallback with limitations and next steps. |
| Invalid citations | Invalid source IDs are removed; sources are rebuilt from evidence metadata. |
| Prompt injection in a page | Content is fenced as untrusted data, never executed as instructions; flagged sources are added to limitations. |
| Missing API key | The corresponding tool fails gracefully instead of crashing the graph. |

Errors and warnings are preserved in the final state and printed with the CLI
trace.

## Hallucination mitigation

- The synthesizer is instructed to use **only** retrieved evidence.
- Every finding must cite source IDs that exist in the evidence set.
- Source URLs and titles are overwritten from retrieved evidence metadata; the
  model cannot invent a URL that survives validation.
- Weak/irrelevant sources do not count toward sufficiency or High confidence.
- Confidence is capped by evidence coverage and ambiguity.
- Exact fallback text is used when no verified evidence exists.
- The evaluation suite includes known-entity and obscure-entity cases.

This is not a proof of correctness. It removes common failure modes and makes
the remaining uncertainty visible, but it does not perform claim-level
entailment.

## Observability

Each run carries an `execution_id` and emits JSON logs from
[`app/utils/logging_config.py`](app/utils/logging_config.py). The CLI prints:

- formatted planning and final response;
- per-node latency and token usage;
- tool call records and attempts;
- errors and warnings;
- assumptions and limitations.

The evaluation runner also records per-case latency, input/output tokens,
evidence counts, iteration counts, errors, warnings, and final answer text in
[`evaluation/evaluation_results.json`](evaluation/evaluation_results.json).

## Setup

### Prerequisites

- Python 3.11+
- A Gemini API key
- A Tavily API key

### Install

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### Configure

Windows:

```bash
copy .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

Set at least:

```dotenv
GEMINI_API_KEY="..."
TAVILY_API_KEY="..."
GEMINI_MODEL="gemini-2.5-flash"
```

Optional operational settings:

```dotenv
MAX_RESEARCH_ITERATIONS=2
MAX_SEARCH_RESULTS_PER_QUERY=4
MAX_PAGES_TO_FETCH=5
REQUEST_TIMEOUT_SECONDS=10.0
LLM_TIMEOUT_SECONDS=45.0
MAX_RETRIES=2
RETRY_BACKOFF_SECONDS=0.5
LOG_LEVEL="INFO"
```

If a model's free-tier quota is exhausted, override it for a run:

```bash
GEMINI_MODEL=gemini-3.1-flash-lite python evaluation/run_evaluation.py
```

## Usage

Ask a question with the CLI:

```bash
python -m app.main "Compare the top 3 open-source vector databases for a startup building RAG products."
```

```bash
python -m app.main "Find 5 Indian B2B SaaS startups in HR tech and summarize their positioning."
```

```bash
python -m app.main "What is the market capitalization of Zyxvubq Technologies in 2024?"
```

The CLI prints the plan trace, tool activity, metrics, and the final structured
response.

## Tests

The deterministic test suite covers schemas, tools, graph transitions,
resilience, retries, prompt-injection fencing, confidence calibration, citation
cleanup, and evidence filtering:

```bash
python -m pytest -q
```

Latest result: **39 passed**.

## Evaluation

Run the five-case evaluation:

```bash
python evaluation/run_evaluation.py
```

Test cases live in [`evaluation/test_cases.json`](evaluation/test_cases.json):

| Case | What it tests | Mode |
|---|---|---|
| TC-01 | Normal comparative research | Live Gemini + Tavily |
| TC-02 | Entity discovery with a requested count | Live Gemini + Tavily |
| TC-03 | Ambiguous / underspecified question | Live Gemini + Tavily |
| TC-04 | Planner, search, and fetch outage | Deterministic fault injection |
| TC-05 | Obscure entity / zero evidence | Deterministic fault injection |

The evaluator checks:

- schema validity;
- citation/URL grounding;
- expected confidence;
- fact-level named entities;
- assumptions when required;
- fallback behavior for low-confidence cases;
- latency, token usage, iterations, errors, and warnings.

Latest real run: **5/5 cases passed**. Full analysis:
[`evaluation/evaluation_report.md`](evaluation/evaluation_report.md).
Machine-readable results:
[`evaluation/evaluation_results.json`](evaluation/evaluation_results.json).

The live cases are non-deterministic. Earlier runs exposed a real weakness in
entity discovery: TC-02 sometimes returned only 2–4 requested companies. The
current system adds a discovery breadth pass, wider search/fetch, a quantity
constraint, and an entity-type qualifier. TC-02 is still the case most sensitive
to live search quality.

## Sample outputs

Three real sample responses are in `examples/`:

- [`sample_output_1.json`](examples/sample_output_1.json) — comparative research, High confidence.
- [`sample_output_2.json`](examples/sample_output_2.json) — entity discovery, High confidence.
- [`sample_output_3.json`](examples/sample_output_3.json) — obscure entity, Low-confidence fallback.

## Design tradeoffs

| Decision | Benefit | Cost |
|---|---|---|
| Single agent with a bounded loop | Simple to reason about and debug | No parallel specialist agents or debate/judge step |
| Two research iterations by default | Bounds latency and API cost | Some questions may need more depth |
| Lexical + search-score relevance | Transparent, cheap, no extra model call | Misses semantic relevance; can mis-rank generic list pages |
| Domain-aware content dedupe | Preserves cross-domain corroboration | Identical content within one domain is dropped even if useful |
| Deterministic confidence rubric | Consistent and testable | Heuristic; does not model contradiction or recency |
| Tavily + fetch instead of a crawler | Fast to build and reliable enough | Search API cost; JS-heavy/paywalled pages may fail |
| Synchronous execution | Easy CLI and state handling | Slower than async tool fan-out |
| CLI instead of a service | Focused on orchestration quality | No HTTP API, auth, or queueing |
| JSON logs + in-state metrics | Easy local tracing | No distributed tracing backend |

## How would this be made production-ready?

1. **Service layer:** wrap the graph in FastAPI, add request validation,
   authentication, per-user rate limits, and async background jobs for long
   research runs.
2. **Durable execution:** persist graph state/checkpoints so a failed run can
   resume rather than restart.
3. **Caching:** cache search/fetch results by canonical URL and query, with TTL
   and provenance metadata.
4. **Concurrency:** run independent searches/fetches concurrently with a global
   deadline and cancellation.
5. **Provider resilience:** circuit breakers, quota-aware model routing,
   fallback providers, and separate timeouts per tool.
6. **Security:** stricter SSRF/redirect allowlists, content-size limits,
   secret management, and injection telemetry.
7. **Evaluation in CI:** deterministic replay fixtures for regression tests plus
   a small scheduled live smoke suite.
8. **Human review:** route Low-confidence or high-stakes answers to review
   rather than presenting them as final.

## What would be monitored in production?

- Request success/error rate and p50/p95/p99 latency.
- Token usage and cost per request, per node.
- Search/fetch error rate and retry rate.
- Empty-evidence rate and iteration distribution.
- Citation validity rate and source-domain diversity.
- Confidence-level distribution and drift.
- Prompt-injection detections.
- Model/provider quota errors and fallback activation.
- Evaluation pass rate over time.

## Known limitations

- **Search quality dominates entity-discovery recall.** The agent cannot name
  five companies if the retrieved sources only describe two.
- **Paywalls and JS-heavy pages** may return placeholders or fail extraction.
- **Lexical relevance is not semantic.** It can miss paraphrases and can
  over-rank query-term matches.
- **Confidence is calibrated by heuristics**, not by claim-level verification.
- **Evaluation is a single live snapshot.** It is not a statistically robust
  benchmark.
- **No memory or follow-up conversation**; each run is independent.
- **No async execution**, so latency is higher than necessary.
- **No non-agentic baseline** yet for direct comparison.

## Future improvements

- Candidate extraction and alias resolution before synthesis for
  "find N" questions.
- Claim-level entailment/NLI scoring against cited snippets.
- Source recency and contradiction-aware confidence.
- Caching and async tool execution.
- A non-agentic baseline and multi-run evaluation distributions.
- Optional FastAPI service and tracing integration (OpenTelemetry/Langfuse).

## Repository layout

```text
app/
  graph.py              LangGraph assembly and conditional routing
  main.py               CLI entrypoint
  schemas.py            Pydantic response and tool schemas
  state.py              LangGraph state definition
  config.py             Environment-backed settings
  nodes/
    planner.py          Plan and follow-up query generation
    researcher.py       Tool selection and execution
    evidence.py         Relevance, dedupe, coverage, sufficiency
    synthesizer.py      Synthesis, citation validation, confidence
  tools/
    web_search.py       Tavily search tool
    fetch_page.py       Safe page fetch/extract tool
  utils/
    llm.py              Structured invocation with retries
    retry.py            Bounded exponential backoff
    sanitize.py         Prompt-injection fencing/detection
    urls.py             URL canonicalization/direct URL extraction
    metrics.py          Per-node latency/token tracking
    logging_config.py   JSON logging
    errors.py           Structured fallback responses
evaluation/
  test_cases.json       Five evaluation cases
  run_evaluation.py     Evaluation runner and assertions
  evaluation_results.json
  evaluation_report.md
examples/
  sample_output_*.json
tests/
  test_graph.py
  test_tools.py
  test_schemas.py
  test_failure_cases.py
  test_resilience.py
```

## Demo

A short demo video accompanies the submission. It shows one happy-path research
run (live planning, search, evidence filtering, synthesis) and one edge case
(TC-05: obscure entity, zero verified evidence, Low-confidence fallback).
