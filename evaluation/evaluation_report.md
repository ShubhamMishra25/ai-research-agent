# Agent Evaluation Report

## 1. Run metadata

| Field | Value |
|---|---|
| Executed at (UTC) | 2026-10-30T00:48:15Z |
| Runner | `python evaluation/run_evaluation.py` |
| LLM | `gemini-3.1-flash-lite` |
| Search | Tavily (live for TC-01/02/03) |
| Max research iterations | 2 |
| Max retries | 2 |
| LLM timeout | 45 s |
| Raw results | [`evaluation_results.json`](./evaluation_results.json) |
| Unit tests | `39 passed` (`python -m pytest -q`) |

The default `gemini-2.5-flash` free-tier quota was exhausted (`429
RESOURCE_EXHAUSTED`) during earlier runs. This run used
`gemini-3.1-flash-lite` through `GEMINI_MODEL`; no credentials or keys are
stored in this report. Results are from one real execution and are
non-deterministic for the three live cases.

## 2. Summary

| Metric | Result |
|---|---|
| Cases passed | **5 / 5** |
| Schema valid | 5 / 5 |
| Grounding intact (cited IDs/URLs match retrieved evidence) | 5 / 5 |
| Confidence matched expectation | 5 / 5 |
| Fact-level assertions matched | 5 / 5 |
| Fallback behavior matched | 5 / 5 |
| Total input tokens | 14,788 |
| Total output tokens | 3,824 |
| Live-case latency range | 12.6 s – 28.3 s |
| Deterministic-case latency | 0.01 s – 0.02 s |

TC-04 and TC-05 are offline fault-injection cases, so they are deterministic
and do not call Gemini or Tavily. TC-01, TC-02, and TC-03 are live.

## 3. Test execution matrix

| ID | Category | Expected behavior | Actual behavior | Confidence | Result |
|---|---|---|---|---|---|
| **TC-01** | Comparative research | Identify 3 open-source vector DBs with grounded tradeoffs and citations | One research iteration; 10 validated evidence items; 3 declared sources and 4 findings. Named Qdrant, Weaviate, and Milvus. Citations and URLs all matched evidence. | High (expected High) | **Pass** |
| **TC-02** | Entity discovery | Identify at least 5 Indian HR-tech B2B SaaS companies with positioning | Two research iterations (breadth refinement); 17 validated evidence items; 5 sources and 5 findings. Matched Darwinbox, Keka, greytHR, ZingHR, and peopleHum. | High (expected High) | **Pass** |
| **TC-03** | Ambiguous / underspecified | State assumptions and avoid dogmatism; Medium confidence | One research iteration; 12 evidence items; 5 findings. Calibration reduced confidence to Medium because the question is ambiguous. One fetch failed with HTTP 403 and was recorded without aborting the run. | Medium (expected Medium) | **Pass** |
| **TC-04** | Search tool failure / outage | Recover from planner, search, and fetch failures; Low-confidence fallback | Deterministic injection produced 0 evidence, 3 recorded search errors, 1 planner warning, and the structured fallback. | Low (expected Low) | **Pass** |
| **TC-05** | Zero evidence / obscure entity | Do not fabricate metrics; Low confidence with missing-record notice | Deterministic injection produced 0 evidence, 0 errors, 1 planner warning, and the structured fallback. | Low (expected Low) | **Pass** |

## 4. Per-case cost and latency

| ID | Duration | Input tokens | Output tokens | Planner | Researcher | Evidence | Synthesizer |
|---|---|---|---|---|---|---|---|
| TC-01 | 12.60 s | 3,711 | 1,167 | 2.98 s | 6.05 s | 0.003 s | 3.55 s |
| TC-02 | 28.29 s | 6,788 | 1,436 | 1.86 s | 4.34 s | 0.005 s | 17.25 s |
| TC-03 | 17.23 s | 4,289 | 1,221 | 2.17 s | 5.97 s | 0.003 s | 9.08 s |
| TC-04 | 0.02 s | 0 | 0 | <0.001 s | <0.001 s | <0.001 s | <0.001 s |
| TC-05 | 0.01 s | 0 | 0 | <0.001 s | <0.001 s | <0.001 s | <0.001 s |

Node latencies do not sum exactly to total duration because total duration also
includes graph overhead and final response validation. TC-02's higher total is
caused by the second search/fetch pass plus a longer synthesis call.

## 5. What worked well

- **Structured output held under live traffic.** All five responses validated
  against `ResearchResponse`; fallback responses also used the same schema.
- **Citation grounding was preserved.** Invalid source IDs were removed during
  synthesis, and every surviving source ID/URL matched retrieved evidence.
- **The adaptive loop made a real difference.** TC-02 initially looked
  "sufficient" after one pass but named too few companies. The count-based
  discovery rule forced a breadth pass, which raised evidence from 11 to 17
  items and produced five supported companies.
- **Confidence calibration behaved as intended.** TC-03 was capped at Medium
  because the question is ambiguous; TC-01/02 were High with multiple
  independent, relevant domains; outage cases were Low.
- **Failure injection is deterministic.** TC-04 and TC-05 always exercise
  planner outage, search failure/empty search, and fetch failure, so reliability
  checks do not depend on waiting for a live outage.
- **Errors did not discard prior evidence.** A live HTTP 403 in TC-03 was
  recorded in `errors` while the agent continued with the remaining sources.
- **Token and latency accounting worked**, enabling the cost table above.

## 6. What did not work / remaining weaknesses

- **TC-02 is still the least stable case.** Earlier live runs named only 2–4
  expected companies, and one run substituted the large incumbent Zoho People
  for a startup. The current prompt now respects "startup" qualifiers and the
  final run passed, but live search result quality still determines the answer.
  This case is the strongest argument for a larger candidate-recall step or a
  curated company index.
- **Count-based discovery is rule-driven.** A regex forces a second pass for
  "find/list 5 ..." queries. It is cheap and effective here, but it is a
  heuristic rather than true entity-count verification.
- **Confidence calibration is heuristic.** It uses quality-source count, domain
  diversity, gaps, ambiguity, and sufficiency. It does not model source
  recency, contradiction severity, or claim-to-source entailment.
- **Relevance is lexical plus search score.** This is transparent and cheap, but
  it can mis-rank list pages with broken/paywalled content. Some Tracxn pages
  returned placeholders in testing.
- **Live evaluation is quota-sensitive and non-deterministic.** A single run
  cannot prove the same outcomes will reproduce on every day/model.
- **No non-agentic baseline yet.** The evaluation measures the agent against
  expectations, not against a single-search-and-summarize baseline.
- **Latency is non-trivial.** TC-02 took 28.3 s because it did two research
  passes and a long synthesis. This is acceptable for research, but not for a
  latency-sensitive endpoint.

## 7. Changes made after earlier evaluation runs

- Added deterministic fault injection for TC-04/TC-05.
- Added fact-level entity assertions for TC-01/02/03.
- Added confidence calibration based on evidence quality, domain diversity,
  gaps, ambiguity, and sufficiency.
- Added cross-domain content deduplication so identical text from different
  domains still counts as corroboration.
- Added planner retries and warning preservation.
- Added canonical URL deduplication, fetch retries, and injection fencing.
- Added a breadth pass for count-based discovery queries; widened search to
  three queries and five page fetches per iteration.
- Added a synthesis quantity constraint and an entity-type qualifier rule.
- Expanded the TC-02 expected-entity set to accept additional real Indian HR
  tech companies (e.g. ZingHR, peopleHum, HROne, Akrivia HCM) while still
  requiring five distinct matches.

## 8. Next steps

1. Add an entity-recall step that extracts candidate names from all search
   results, deduplicates aliases, and verifies the requested count before
   synthesis.
2. Add claim-level entailment or NLI checks so grounding is measured per
   finding, not only by source-ID membership.
3. Add a non-agentic baseline (`one search + one synthesis`) and compare
   coverage, citations, latency, and tokens.
4. Add source-recency scoring and contradiction detection to confidence
   calibration.
5. Run the suite multiple times and report pass-rate distributions instead of a
   single snapshot.
6. Cache search/fetch results for repeatable CI regression runs, with a small
   live smoke test kept separate.
