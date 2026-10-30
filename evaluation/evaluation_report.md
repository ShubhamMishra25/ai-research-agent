# Agent Evaluation Report

## 1. Run metadata

| Field | Value |
|---|---|
| Executed at (UTC) | 2026-10-30T00:22:09Z |
| Runner | `python evaluation/run_evaluation.py` |
| LLM | `gemini-2.5-flash-lite` |
| Search | Tavily (live) |
| Max research iterations | 2 |
| Max retries | 2 |
| Raw results | [`evaluation_results.json`](./evaluation_results.json) |

The first attempt used `gemini-2.5-flash`; it hit the free-tier daily request
quota (`429 RESOURCE_EXHAUSTED`), so the successful run was executed with
`gemini-2.5-flash-lite` via `GEMINI_MODEL`. The evaluation is live and
non-deterministic; results below reflect one real execution.

## 2. Summary

| Metric | Result |
|---|---|
| Schema valid | 5 / 5 |
| Sources present | 5 / 5 |
| Grounding intact (cited IDs/URLs match retrieved evidence) | 5 / 5 |
| Confidence matched expectation | 2 / 5 |
| Hard fallback matched expectation | 3 / 5 |
| Total input tokens | 25,235 |
| Total output tokens | 13,301 |
| Latency range | 17.0 s – 41.2 s |

## 3. Test execution matrix

| ID | Category | Expected behavior | Actual behavior | Confidence | Result |
|---|---|---|---|---|---|
| **TC-01** | Comparative research | Identify 3 open-source vector DBs with grounded tradeoffs and citations | Structured comparison produced; 8 sources, 6 findings; all citations grounded. Confidence was **High** as expected. | High (expected High) | **Pass** |
| **TC-02** | Entity discovery | Identify 5 Indian HR-tech B2B SaaS startups with positioning | Structured answer produced; 8 sources, 6 findings. Only 2 of 8 sources cleared the relevance threshold, so the model returned **Medium** instead of High. | Medium (expected High) | **Partial — confidence mismatch** |
| **TC-03** | Ambiguous / underspecified | State assumptions and avoid dogmatism; Medium confidence | Loop ran 2 iterations (15 evidence items, 9 findings) and the answer was well grounded, but confidence was **High**. The agent did not treat ambiguity as a confidence reducer. | High (expected Medium) | **Partial — overconfident** |
| **TC-04** | Search tool failure / outage | Recover from tool errors; return Low confidence fallback | Search did **not** fail during this run, so this case did not exercise the outage path. The agent ran 2 iterations, gathered 13 sources, and returned **High** confidence. | High (expected Low) | **Fail — invalid test condition** |
| **TC-05** | Zero evidence / obscure entity | Avoid fabricating metrics; Low confidence with missing-record notice | Returned **Low** confidence and 16 sources, but did not use the deterministic “Unable to complete research” fallback. The evaluator therefore marked the hard-fallback assertion false. | Low (expected Low) | **Partial — fallback assertion mismatch** |

## 4. Per-case cost and latency

| ID | Duration | Input tokens | Output tokens | Planner | Researcher | Evidence | Synthesizer |
|---|---|---|---|---|---|---|---|
| TC-01 | 16.97 s | 3,108 | 2,350 | 2.03 s | 7.08 s | 0.002 s | 7.84 s |
| TC-02 | 19.25 s | 3,583 | 2,285 | 2.02 s | 5.81 s | 0.002 s | 11.39 s |
| TC-03 | 21.82 s | 5,553 | 3,052 | 1.73 s | 6.17 s | 0.002 s | 8.88 s |
| TC-04 | 41.17 s | 5,544 | 2,672 | 2.34 s | 14.21 s | 0.013 s | 14.75 s |
| TC-05 | 26.32 s | 7,447 | 2,942 | 1.80 s | 8.14 s | 0.011 s | 9.02 s |

## 5. What worked well

- **Structured output held under live traffic.** All five responses validated against `ResearchResponse`; no malformed JSON or missing fields reached the caller.
- **Citation grounding survived synthesis.** No invalid source IDs/URLs survived validation in any case. Findings without a declared source were dropped rather than shown.
- **The adaptive loop engaged when needed.** TC-03, TC-04, and TC-05 required a second research cycle. On TC-04 the second cycle increased evidence from 7 to 13 sources and flipped sufficiency.
- **Tool selection was respected.** TC-03's planner selected search-only, and the researcher skipped page fetching entirely.
- **Failure handling degraded safely.** During the quota-exhausted first attempt, the agent still produced schema-valid Low-confidence responses instead of crashing.
- **Token and latency accounting worked**, enabling the cost table above.

## 6. What did not work

- **Confidence calibration is the main weakness.** TC-02 was under-confident (Medium vs High), while TC-03 was over-confident (High vs Medium). The model's confidence is not calibrated against question ambiguity or source-quality thresholds.
- **TC-04 does not actually test an outage.** A live Tavily call usually succeeds, so the expected Low-confidence path is not exercised. This case needs fault injection (invalid key, mocked timeout/500) rather than hoping for a failure.
- **TC-05 missed the deterministic fallback assertion.** It returned Low confidence but synthesized from noisy evidence, so the exact fallback string was absent. Either the assertion should check “Low + limitations + no fabricated metric” or the case should force zero evidence.
- **Relevance filtering is conservative.** TC-02 kept 8 sources but only 2 were quality sources; TC-03 kept 15 but only 1 cleared the threshold. This suppresses confidence without necessarily improving the answer.
- **Latency is high.** 17–41 s per query, with TC-04 taking 41 s due to retries plus a second research cycle.
- **Evaluation is not reproducible offline.** Live search and live LLM calls make results non-deterministic and quota-sensitive.

## 7. Next steps

1. Add deterministic fault injection for TC-04/TC-05 (mock Tavily failures and empty evidence) so failure paths are always tested.
2. Add fact-level assertions (e.g. three named vector DBs, five named HR-tech companies) instead of only schema/grounding checks.
3. Replace the boolean confidence match with a rubric: ambiguity handling, source diversity, corroboration, and evidence recency.
4. Improve confidence calibration using evidence coverage, number of independent quality domains, and explicit ambiguity signals from the planner.
5. Persist final response text per case in `evaluation_results.json` (now included by the runner) and add a non-agentic baseline for comparison.
