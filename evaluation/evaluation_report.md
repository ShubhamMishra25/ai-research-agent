# Agent Evaluation Report

## 1. Overview & Test Battery
The evaluation suite exercises five boundary conditions:
1. **TC-01 (Comparative Research)**: Evaluates structured comparison between technical products.
2. **TC-02 (Entity Discovery)**: Evaluates extraction and positioning synthesis for five regional B2B entities.
3. **TC-03 (Ambiguous Query)**: Evaluates handling of an underspecified prompt requiring explicit assumptions.
4. **TC-04 (Tool Failure / Outage)**: Verifies the fallback path when search tool fails or times out.
5. **TC-05 (Zero Ground Truth)**: Verifies that nonexistent entities result in honest uncertainty and Low confidence.

## 2. Test Execution Matrix

The matrix below is a planned/example result summary, not execution evidence.
Run `python evaluation/run_evaluation.py` to execute the live evaluation. That
command writes the timestamped, machine-readable results to
[`evaluation_results.json`](./evaluation_results.json). The evaluation uses
external Gemini and Tavily services, so results are not fully offline or
deterministic.

| Test ID | Category | Expected Behavior | Actual Behavior | Status |
|---|---|---|---|---|
| **TC-01** | Comparative Research | Identification of 3 DBs with pros/cons and citations | Not executed | **Illustrative — not executed** |
| **TC-02** | Entity Discovery | 5 Indian HR Tech startups with positioning | Not executed | **Illustrative — not executed** |
| **TC-03** | Ambiguous Query | Explicit assumptions, Medium confidence | Not executed | **Illustrative — not executed** |
| **TC-04** | Tool Outage | Deterministic fallback, Low confidence, no crash | Not executed | **Illustrative — not executed** |
| **TC-05** | Obscure Entity | Refusal to fabricate facts, Low confidence | Not executed | **Illustrative — not executed** |
