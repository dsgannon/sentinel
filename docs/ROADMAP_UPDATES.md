# Sentinel Roadmap Updates

Supplements `sentinel_development_plan.pdf`. Documents decisions made during development
that extend or modify the original plan.

---

## Phase 6 Updates — Gen AI Layer

### Original Plan
Claude via Amazon Bedrock for conceptual review, auto-documentation, and regulatory RAG.

### Updates

#### 1. Start with Claude API Directly, Swap to Bedrock in Phase 4
The LLM logic (prompt engineering, RAG, tool use) will be built against the `anthropic`
Python SDK first. Swapping to Bedrock is a client-level change — same Claude model,
different endpoint. This removes AWS setup friction during the learning phase.

#### 2. Agentic Validation Workflow
Rather than Claude answering questions in isolation, the target architecture is a
**validation agent** that orchestrates Sentinel's metric functions via tool use.

Claude is given tools like:
- `compute_auc(y_true, y_pred)`
- `check_fairness(y_pred, sensitive_attr)`
- `check_drift(expected, actual)`
- `run_stress_test(X, y_true, mask)`

The agent decides which tools to call, interprets the results, and produces structured
findings — mimicking how a human validator would work through a model review.

This maps directly to the Anthropic Gen AI Developer cert objectives.

#### 3. Human-in-the-Loop Review (Required)
AI findings are **drafts**, not decisions. The workflow is:

1. Claude runs quantitative checks and drafts findings with severity flags
2. Human validator reviews each finding and marks it as:
   - `confirmed_risk` — accepted, goes into the regulatory file
   - `acceptable` — risk acknowledged but within tolerance
   - `false_positive` — Claude was wrong, overridden
3. Human decision + rationale logged to the audit trail with timestamp and user

This is SR 11-7 compliant — human accountability is preserved. Claude assists,
humans decide.

#### 4. LLM Evaluation Metrics
Claude's outputs are evaluated both quantitatively and by humans:

**Quantitative (for development and testing):**
- **Faithfulness** — do Claude's findings accurately reflect the data provided?
- **Consistency** — does Claude produce the same finding on the same input across runs?
- **Coverage** — does Claude flag all issues a human validator would catch?
  Measured against a labeled test set of known model problems.

**Human evaluation (for production):**
- **Human override rate** — percentage of Claude findings overridden by validators.
  High override rate signals prompt quality issues. Tracked over time.
- **Structured scoring rubric** — validators rate each finding on accuracy,
  completeness, and actionability.

The override rate is particularly valuable for SR 11-7 documentation — it provides
evidence that humans are genuinely reviewing AI output, not rubber-stamping it.

---

## Phase 6 Build Order

1. **Conceptual soundness review** — prompt engineering, structured output, no RAG
2. **Auto-documentation** — validation results to narrative sections
3. **Agentic validator** — tool use, Claude orchestrates metric functions
4. **Human review interface** — Streamlit UI for finding review + override logging
5. **LLM evaluation** — faithfulness, consistency, override rate tracking
6. **Regulatory RAG** — vector DB over SR 11-7, OCC 2011-12, CFPB guidance
7. **Bedrock migration** — swap `anthropic` client to `boto3` Bedrock

---

## Architecture Decisions

### LLM Outputs Are Structured, Not Free Text
All Claude outputs use structured JSON so they can be:
- Displayed in the review UI field by field
- Stored in the database
- Evaluated programmatically
- Compared across runs for consistency

### Audit Trail Integration
Every Claude finding and every human override is written to the existing `audit_log`
table via `log_action()`. The regulatory file contains both.

### No Fully Automated Decisions
Claude never approves or rejects a model. It produces findings. Humans decide.
This is non-negotiable for SR 11-7 compliance.
