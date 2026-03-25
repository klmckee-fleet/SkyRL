# Hint Augmentation & RLTF-SD: Limitations and Directions

## Setup

We RL-train Qwen3.5-35B-A3B with GRPO on tool-use tasks across 12 SaaS environments. Each task gives the model a prompt, a set of MCP tools, and a programmatic verifier. The model runs a multi-turn loop — generating tool calls, receiving observations, iterating — until it signals `<done>` or hits a context/turn limit. The verifier then scores the final environment state.

GRPO requires reward variance within each prompt group to compute a learning signal. With `n_samples_per_prompt=8`, if all 8 rollouts score 0 the prompt contributes zero gradient. Many tasks are hard enough that this happens frequently — 37% of training prompts across steps 1–19 had all 8 samples score 0.

**Hint augmentation** was introduced to rescue these dead prompts. When all samples fail, the mechanism:
1. Takes the best failed rollout's verifier output
2. Parses `ERROR_ACCUMULATOR` / `SUCCESS_ACCUMULATOR` markers from the verifier's stdout using regex — no LLM involved
3. Formats them into a hint string prepended to the task prompt
4. Launches 2 new rollouts with the hint injected

The goal is to unlock at least one successful rollout per dead prompt, creating the reward variance GRPO needs.

**RLTF-SD** (Reward from Teacher Feedback, Student Distribution) is applied to the successful hinted rollouts. Because the hinted rollout was conditioned on `[prompt + hint]` but the model at inference time only sees `[prompt]`, you cannot naively train on the hinted trajectory — you would be teaching the model to produce behavior that requires information it won't have. RLTF-SD's answer: swap the `prompt_ids` of the hinted trajectory with the original bare-prompt `prompt_ids` before the gradient update, so the loss is computed as `log π(y_hint | x_bare)`. The intended effect is that the model learns to produce hint-quality outputs from the original prompt alone.

---

## Problem 1: Most hints are no-ops (context overflow)

**70% of hints are** `"Checks failed (1): [X] No final_answer provided (None)"`.

The verifier short-circuits when `final_answer=None`. It never reaches its check logic. Budget env is the main driver: 50K tokens of tool definitions alone (53% of the 96K context window), leaving only 45K for actual work. 71% of budget trajectories hit `max_input_length` before the model can call `final_answer`.

The hint is not wrong — it is the verifier faithfully reporting that nothing was submitted. It tells the model nothing it didn't already experience.

From steps 1–19 (112 hinted prompts):

| category | count | % | description |
|---|---|---|---|
| no_answer | 65 | 58% | verifier got `final_answer=None` — context overflow before submission |
| fallback | 21 | 19% | no verifier stdout — static string "try a different approach" |
| informative | 26 | 23% | real check failures with specific expected vs. actual values |

Context overflow is a separate problem that needs its own solution (tool pruning, larger context, smarter truncation). Until it is solved, the majority of the hint budget is wasted before it starts.

---

## Problem 2: Informative hints are still incomplete

For the 23% of hints that contain real check failures, the hint may still be insufficient to act on. Consider:

```
Checks failed (1): [X] F1=0: no correct rows (expected 3, predicted 3)
```

This tells the agent the right number of rows was returned but none matched. It does not say:
- Which table was queried
- What the actual row values were
- What filter produced them
- Whether the mistake was a wrong column name, wrong value format, wrong date range, or a schema assumption mismatch

The agent has no transcript of what it did. The hint is a grade, not a diagnosis.

**The minimal useful unit of feedback is `[action, observation, hint]`, not `[hint]` alone.** Without knowing what action produced the failing observation, the hint reduces to "you got it wrong — try something different."

A harder version of this problem: many failures require a **debug or search session** to produce a useful hint at all. The lesson is the search itself — `SELECT DISTINCT`, `PRAGMA table_info`, checking column types — not a sentence you can write before the search. For tool-specific failures involving schema quirks, API encodings, or undocumented constraints, no pre-attempt hint is possible in principle. The fix requires discovering facts about the environment that only exist inside the tool.

---

## Problem 3: Hints that are specific still may not transfer

Even a precise hint like `"You created a booking but with the wrong guest count"` may not help:

- The agent doesn't know *when* in the trajectory it made that mistake
- It doesn't know *why* — a misread of the prompt, a wrong API argument, a tool that silently accepted an invalid value?
- Random variation between rollouts means the mistake may not even recur in the same form

Most informative hints are closer to **"don't screw it up"** than to actionable guidance. They identify that something was wrong without giving the agent the context to understand where its reasoning diverged.

---

## Problem 4: Hints need to be lessons, not error reports

The current hint is a raw verifier error message. What would be more useful is a **lessons-learned reformulation** — a generalizable strategy derived from the specific failure, produced by an LLM that reads the full transcript and verifier output together.

**Raw verifier hint:**
```
Checks failed (1): [X] F1=0: no correct rows (expected 3, predicted 3)
```

**With the tool call history visible:**
```
[Turn 3] Action: SELECT merchant_name, amount, date FROM transactions
                 WHERE category = 'groceries' ORDER BY date DESC LIMIT 3
  Observation: [{"merchant_name": "Whole Foods", "amount": 45.20, ...}]
```

**Lessons-learned hint:**
```
Your query structure was correct but you filtered on the wrong column value.
Before writing your final query, explore the distinct values in the columns
you plan to filter on — the actual category strings in the database may not
match what the task description says. Use a SELECT DISTINCT query first to
verify the exact values before filtering.
```

The lessons-learned version teaches a reusable strategy. The raw version is a grade. This requires an LLM call during training — the current implementation deliberately avoids this for latency and cost reasons.

---

## RLTF-SD: applicability to tool use

RLTF-SD was developed in a reasoning context where "tools" are general problem-solving steps — logical moves the model reasons through. In that setting, a hint about reasoning quality transfers naturally: the model internalizes a better strategy and applies it from the bare prompt. The hint describes a conceptual mistake, and the fix is a conceptual correction.

**Tool use breaks this assumption in two ways.**

**1. Tool calls produce privileged information the model cannot consolidate generally.**

A gradient that improves SQL filtering on one database does almost nothing to improve SQL filtering on a different database. The successful hinted behavior was correct because it happened to match what *this specific tool* does — the schema, column names, value encodings, table structure. Training on it teaches facts about one environment, not a transferable skill. The only generalizable bridge between environments is **reasoning about tools** — how to explore an unknown schema, how to interpret an unexpected result, how to recover from a tool error. Hints should focus on where that reasoning went wrong, not on reproducing the specific corrected action.

**2. Hint-specific reasoning creates incoherent amortized behavior.**

The hinted rollout directly references the hint. The agent may generate reasoning like:

> *"Previously, row 3 had incorrect values — I'll avoid using `category='groceries'` this time and check distinct values first..."*

When RLTF-SD trains this trajectory under the bare prompt, the model is asked to produce that reasoning from a context where it never saw the hint. At inference time the model has been trained to reason as if it had information it doesn't have — producing responses that reference prior events that didn't happen. This is not just off-policy, it is incoherent as a general behavior.
