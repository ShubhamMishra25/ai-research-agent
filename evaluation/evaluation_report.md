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
| **TC-01** | Comparative Research | Identification of 3 DBs with pros/cons and citations | Returned Qdrant, Milvus, and Chroma with citations `[S1]`, `[S2]` | **PASSED** |
| **TC-02** | Entity Discovery | 5 Indian HR Tech startups with positioning | Discovered Darwinbox, Keka, greytHR, Zimyo, Leena AI | **PASSED** |
| **TC-03** | Ambiguous Query | Explicit assumptions, Medium confidence | Declared developer stack assumptions; compared LangGraph, AutoGen, CrewAI | **PASSED** |
| **TC-04** | Tool Outage | Deterministic fallback, Low confidence, no crash | Search failed gracefully; returned structured fallback with next steps | **PASSED** |
| **TC-05** | Obscure Entity | Refusal to fabricate facts, Low confidence | Emitted Low confidence; stated zero verifiable records discovered | **PASSED** |
