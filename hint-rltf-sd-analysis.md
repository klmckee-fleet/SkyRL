  # Hint Augmentation & RLTF-SD: Limitations and Directions

  ## Setup

  We RL-train Qwen3.5-35B-A3B with GRPO on tool-use tasks across 12 SaaS environments. Each task gives the model a prompt, a set of MCP tools, and a programmatic verifier. The model runs a multi-turn loop — generating tool calls, receiving observations, iterating — until it signals `<done>` or hits a context/turn limit. The verifier then scores the final environment state.

  GRPO requires reward variance within each prompt group to compute a learning signal. With `n_samples_per_prompt=8`, if all 8 rollouts score 0 the prompt contributes zero gradient. Many tasks are hard enough that this happens frequently — 37% of training prompts across steps 1–19 had all 8 samples score 0.

  **Hint augmentation** was introduced to rescue these dead prompts. When all samples fail, the mechanism takes the best failed rollout's verifier output, parses `ERROR_ACCUMULATOR` / `SUCCESS_ACCUMULATOR` markers from the verifier's stdout using regex — no LLM involved — formats them into a hint string prepended to the task prompt, and launches 2 new rollouts with the hint injected. The goal is to unlock at least one successful rollout per dead prompt, creating the reward variance GRPO needs.

  **RLTF-SD** (Reward from Teacher Feedback, Student Distribution) is applied to the successful hinted rollouts. Because the hinted rollout was conditioned on `[prompt + hint]` but the model at inference time only sees `[prompt]`, you cannot naively train on the hinted trajectory — you would be teaching the model to produce behavior that requires information it won't have. RLTF-SD's answer: swap the `prompt_ids` of the hinted trajectory with the original bare-prompt `prompt_ids` before the gradient update, so the loss is computed as `log π(y_hint | x_bare)`. The intended effect is that the model learns to produce hint-quality outputs from the original prompt alone.

  ---

  ## Problems with hints

  Besides context overflow, problems center on whether hints are actually useful, given both their base content and strategy for incorporation.

  **No-op hints** 70% of hints are `"Checks failed (1): [X] No final_answer provided (None)"`. The verifier short-circuits when `final_answer=None` and never reaches its check logic. Budget env is the main driver: 50K tokens of tool definitions alone (53% of the 96K context window), leaving only 45K for actual work. 71% of budget trajectories hit `max_input_length` before the model can call `final_answer`. From steps 1–19 (112 hinted prompts): 65 no_answer (58%), 21 fallback (19%), 26 informative (23%). Context overflow needs its own solution — until it is solved, the majority of the hint budget is wasted before it starts.

  **Incomplete hints** For the 23% of hints that contain real check failures, the hint may still be insufficient to act on. `"F1=0: no correct rows (expected 3, predicted 3)"` tells the agent something was wrong but not which table, what the actual values were, what filter produced them, or what kind of mistake caused it. The agent has no transcript of what it did. The hint is a grade, not a diagnosis. The minimal useful unit of feedback is `[action, observation, hint]`, not `[hint]` alone.

  **Untransferable hints** For tool-specific failures involving schema quirks, API encodings, or undocumented constraints, no pre-attempt hint is possible in principle. The lesson is the search itself — `SELECT DISTINCT`, `PRAGMA table_info`, checking column types — not a sentence you can write before the search. The fix requires discovering facts about the environment that only exist inside the tool.

  **Hints that require search or debug** Even a precise hint like `"You created a booking but with the wrong guest count"` doesn't tell the agent when in the trajectory it made that mistake, why, or how to find it again given random variation between rollouts. Most informative hints are closer to "don't screw it up" than to actionable guidance.

  ---

  ## Problems for RLTF-SD

  Most problems are going to come from the fact that tools correspond to specific computations that the model cannot or should not try to approximate. RLTF-SD originally concerns the quality of reasoning, so the "tools" are more like general conceptual turns and problem solving strategies.
  As a general statement, trial-and-error is not usually a great way to learn software anyways, compared to reading the documentation and reasoning about what the tool is expected to do, given the args and the context.

  **Tool calls produce privileged information the model cannot consolidate generally.** A gradient that improves SQL filtering on one database does almost nothing on a different database. The successful hinted behavior was correct because it matched what *this specific tool* does — the schema, encodings, table structure. Training on it teaches facts about one environment, not a transferable skill. The generalizable bridge is reasoning about tools: how to explore an unknown schema, how to interpret unexpected results, how to recover from errors. Hints should target where that reasoning went wrong, not reproduce the specific corrected action.

  **Hints fire hardest on the tasks where off-policy mismatch is worst.** Hints only trigger when all 8 raw samples score 0 — meaning the model's current policy is maximally far from the solution. These are exactly the tasks where `π(y_hint | x_bare)` is smallest, the importance ratio is largest, and RLTF-SD is least stable. The mechanism applies its most aggressive off-policy correction to the examples where it's least likely to work.

  **Hint-specific reasoning creates incoherent amortized behavior.** The hinted rollout directly references the hint — the agent may reason: *"Previously, row 3 had incorrect values — I'll check distinct values first..."* When RLTF-SD trains this under the bare prompt, the model is asked to produce reasoning that references prior events that didn't happen. At inference time it has been trained to reason as if it had information it doesn't have, producing incoherent reasoning chains.

  ---

  ## Paths forward

  In general, the feedback-conditioned trajectory needs to be (1) not conditioned on privileged tool information, (2) on or almost on-policy, (3) more likely to produce task rewards in general.

  **Generate lessons rather than error reports.** Rather than parsing raw verifier output, use an LLM that reads the full transcript and verifier output together to produce a generalizable strategy. Instead of `"F1=0: no correct rows"`, produce: *"Before filtering, verify exact column values with SELECT DISTINCT — the task description may use different terminology than the schema."* The raw version is a grade. The lessons version teaches a reusable strategy. This requires an LLM call during training, which the current implementation deliberately avoids for latency and cost reasons.

  **Include prior trajectory information or records along with the hint.** The agent needs `[action, observation, hint]` to reason about what to change, not `[hint]` alone. The `kevin/0` branch on the `klmckee-fleet/SkyRL` fork begins to address this by injecting a formatted summary of the failed rollout's tool calls and observations (thinking blocks stripped, messages capped at 500 chars, max 10 turns) before the hint text.

  **Restrict hint augmentation to environments where it can work.** Hints are structurally uninformative for budget and reddit (context overflow, LLM-judge verifiers without structured output). Applying hint augmentation selectively to environments with code-based verifiers (carlisle, booking, ticketmaster) might concentrate the signal where it is actually informative.