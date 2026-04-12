# Failure Analysis — Consilium Evaluation History

A forensic record of eval failures by phase. Maintained as a first-class artefact alongside the result JSON files.

---

## Phase 6 — 76.7% Completion (23/30) — RESOLVED

**Eval file:** `phase6_final_2026-04-07_08-24-18.json`

### Failing cases

| Case ID | Category | Latency | Failure mode |
|---------|----------|---------|--------------|
| `aud-003` | Audit | 45,732ms | `analyst_fallback` — JSONDecodeError |
| `amb-003` | Ambiguous | 23,727ms | `analyst_fallback` — JSONDecodeError |
| `edge-002` | Edge | 22,175ms | `analyst_fallback` — JSONDecodeError |
| `risk-002` | Risk | 14,455ms | `analyst_fallback` — JSONDecodeError |
| `aud-004` | Audit | 11,213ms | `analyst_fallback` — JSONDecodeError |
| `rev-007` | Revenue | ~8,000ms | `analyst_fallback` — JSONDecodeError |
| `amb-004` | Ambiguous | ~7,500ms | `analyst_fallback` — JSONDecodeError |

All 7 failures shared the same failure mode: `confidence=0.30`, `fallback_events=["analyst"]`.

### Diagnostic timeline

**Run 1 (Phase 6 first eval, 15s inter-case delay):** 24/30 (80%) — assumed rate limiting. Added delay.

**Run 2 (Phase 6, 30s inter-case delay):** 24/30 — same cases, same failure mode. Rate limiting hypothesis eliminated.

**Wrong hypothesis #2:** Tenacity retry loop was calling `create_llm_client()` once before the loop, meaning all 3 retries used the same `ChatGroq` instance (same API key). Fixed — key rotates per retry now. Eval: still 24/30.

**Correct diagnosis:** Read the actual error string from the API log — not a `429`, not a network error:

```
LLM returned invalid JSON: Unterminated string starting at: line 32 column 53 (char 8664)
LLM returned invalid JSON: Expecting ',' delimiter: line 11 column 317 (char 2930)
```

HTTP 200 responses. `JSONDecodeError` on the response body. The LLM's output was being truncated mid-string by Groq's output token window.

**Why those cases?** Quaestor returns 15–20 large chunks for broad audit/edge/ambiguous queries. `_build_prompt()` fed all chunks with no cap. The system prompt said "one finding per chunk" — so 15 chunks produced 15 findings, generating 8000–10000+ character JSON output. Groq's output token window cut it off mid-string.

### Three-part fix (Phase 7)

**1. Input cap** (`analyst.py`):
```python
_MAX_ANALYST_CHUNKS: int = 6
_MAX_CHUNK_CHARS: int = 600

capped_chunks = chunks[:_MAX_ANALYST_CHUNKS]
text = chunk.chunk_text[:_MAX_CHUNK_CHARS]
```

**2. Output constraint** (system prompt changed from "one finding per chunk" to):
```
Output MAXIMUM 3 findings — prioritise the highest-risk items
finding: 20-100 chars MAXIMUM — one concise sentence only
```

**3. Partial JSON recovery** (`_parse_llm_response()`):
```python
except json.JSONDecodeError:
    start = text.find("[")
    last_complete = max(text.rfind("},"), text.rfind("}\n"))
    if last_complete > start:
        repaired = text[start : last_complete + 1] + "]"
        raw = json.loads(repaired)
        if isinstance(raw, list) and raw:
            return [ComplianceFinding.model_validate(item) for item in raw]
```

### Phase 7 result

**Eval file:** `phase7_final_30_2026-04-10_05-57-30.json`

All 30/30 cases pass. P50=2055ms, P95=2895ms. Gate: PASS.

The six previously failing cases:

| Case ID | Phase 6 latency | Phase 7 latency | Phase 7 result |
|---------|----------------|----------------|----------------|
| `aud-003` | 45,732ms | ~2,100ms | ✅ Pass, confidence=0.85 |
| `amb-003` | 23,727ms | ~1,900ms | ✅ Pass, confidence=0.85 |
| `edge-002` | 22,175ms | ~2,300ms | ✅ Pass, confidence=0.85 |
| `risk-002` | 14,455ms | ~2,000ms | ✅ Pass, confidence=0.85 |
| `aud-004` | 11,213ms | ~1,800ms | ✅ Pass, confidence=0.85 |
| `rev-007` | ~8,000ms | ~2,500ms | ✅ Pass, confidence=0.85 |

---

## Lessons

**1. Structured error attributes on spans beat log parsing.**
The diagnostic required reading a plain-text log file. A `json_parse_error` attribute on the analyst OTel span would have surfaced the pattern in Phoenix immediately — on the first failing run.

**2. "One X per Y" instructions scale with Y.**
"One finding per chunk" × 15 chunks = 15 findings = large JSON. Any instruction that scales with input size needs an explicit ceiling. This should have been in the system prompt from Phase 1.

**3. Phase gates surface real defects.**
Without the ≥65% completion gate, the truncation bug would have shipped silently. The gate is what made it visible and forced resolution before merging.

**4. Wrong hypotheses waste runs.**
Two full eval runs (15s and 30s inter-case delay) were spent on the rate-limiting hypothesis. The correct diagnosis was one `grep` away in the API log.

---

## Visual Analysis

Charts comparing Phase 6 vs Phase 7 per-case latency and confidence distribution:

- [`notebooks/comparison.ipynb`](../../notebooks/comparison.ipynb) — reproducible from stored JSON files
- [`docs/figures/act3_per_case_scatter.png`](../../docs/figures/act3_per_case_scatter.png) — latency scatter with outliers annotated
- [`docs/figures/act4_before_after.png`](../../docs/figures/act4_before_after.png) — before/after confidence + metrics
