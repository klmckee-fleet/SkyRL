# Task-Gen RL Training Fixes Log

Tracking all fixes applied to the multi-turn task generation RL training pipeline.

**Branch**: `deniz/multi-turn-task-gen` on `fleet-ai/SkyRL`
**Pod**: Task-Gen pod (8x H200 RunPod)
**Model**: Qwen/Qwen3.5-9B

---

## Fix #1: Date Awareness + Complexity Guidance + Turn Urgency
**Commit**: `e9d9a68e`
**Symptom**: 100% zero rewards across all trajectories in first training run
**Root Cause**: Model generated tasks with past dates (rejected by booking/ticketmaster environments), trivially simple tasks, and didn't use all available turns
**Fix**:
- Injected `CURRENT_DATE` into system prompt with bolded warning
- Added complexity guidance (aim for 2-8 tool calls)
- Added turn urgency nudge (reminds model to submit before max_turns)

---

## Fix #2: Remove Broken Environments
**Commit**: `75e2b90d`
**Symptom**: forums-homes (Zillow 404s) and wallst (Excel bugs, empty dataroom) environments causing failures
**Root Cause**: Broken upstream environment instances
**Fix**: Removed `forums-homes` and `wallst` from `ENV_KEYS`

---

## Fix #3: Verifier Template + Exploration Gate + Accumulator Patterns
**Commit**: `00178315`
**Symptom**: All harness eval scores 0.0; training collapse (model skips DB exploration, outputs `<task>` immediately); useless hints ("The previous attempt failed")
**Root Causes**:
1. **Wrong DB IDs**: Generated verifiers hardcode wrong user_id (e.g., `user_id=8` instead of `1` for Kenneth Johnson). Model doesn't explore DB to learn correct IDs.
2. **Missing accumulator patterns**: Generated verifiers lack `ERROR_ACCUMULATOR`/`SUCCESS_ACCUMULATOR` stdout markers → `_build_hint_text()` falls back to generic message.
3. **Training collapse**: Model learned to skip exploration and immediately output `<task>`, producing trivial/wrong tasks.
**Fix**:
- Added comprehensive verifier template with `validate_task()` signature, `find_new_entries()` helper, `TASK_FAILED_SCORE`/`TASK_SUCCESSFUL_SCORE` constants
- Added accumulator print patterns (using simple `print()` calls)
- Added "NEVER hardcode database IDs" rule
- Added exploration gate: reject `<task>` if no `describe_db`/`query_db` calls made
- Updated verifier sandbox to accept both `verify()` and `validate_task()` function names
- Increased `MAX_AST_NODES` from 300 to 500

---

## Fix #4: env.env_variables=None Crash + Hint Error Extraction
**Commit**: (pending)
**Symptom**: Still all harness scores 0.0 after Fix #3. Useless hints persist.
**Root Causes**:
1. **`env.env_variables` is `None` in Fleet harness verifier runtime**: The verifier template instructed model to use `env.env_variables["LOGGED_IN_USER"]`, but Fleet harness runtime doesn't populate this field. ALL verifiers using it crash with `TypeError: 'NoneType' object is not subscriptable`.
2. **Verifier errors not extracted for hints**: `_extract_job_results()` only captured `stdout`. When verifiers crash, there's no stdout → hint builder returns generic "The previous attempt failed."
3. **Datetime timezone mismatch**: Verifiers do exact string comparison for datetimes (`"2025-08-08T14:00:00" != "2025-08-08T14:00:00Z"`) → false negatives even when agent completes task correctly.
**Fix**:
- Changed system prompt: embed env var values as string constants in verifiers instead of using `env.env_variables` API
- Added explicit rule: "NEVER use `env.env_variables`"
- Added timezone-tolerant comparison guidance
- `_extract_job_results()` now returns `(score, stdout, error)` tuples — captures stderr/error from crashed verifiers
- `_evaluate_task()` passes error to `_build_hint_text()` so hints include actual crash tracebacks

**Evidence**:
- Job `aff998ed` (reddit): 4/4 sessions crashed with `TypeError: 'NoneType' object is not subscriptable` on `env.env_variables["LOGGED_IN_USER"]`
- Job `fca6c57e` (outlook): 3/4 scored 0.0 due to `"2025-08-08T14:00:00Z" != "2025-08-08T14:00:00"` timezone suffix mismatch

---

## Fix #5: Dict Access Pattern + Error Extraction Path
**Commit**: `973e471f`
**Symptom**: Verifiers crash with `AttributeError: 'dict' object has no attribute 'id'`; error not extracted for hints
**Root Causes**:
1. **Dict vs object access**: Model uses `row.column` (dot notation) but DB queries return Python dicts. Must use `row["column"]`.
2. **Unsupported query methods**: Model invents `.like()`, `.gt()` etc. Only `.eq()`, `.neq()`, `.select()`, `.all()`, `.first()`, `.count()` exist.
3. **Wrong error extraction path**: Code checked `verifier_execution.stderr` (always None). Actual error is at `verifier_execution.result.error.traceback`.
**Fix**:
- Verifier API docs: explicit "rows are dicts, use row['col']"
- Listed all supported query methods, banned invented ones
- Added Python filtering example for unsupported operations
- Fixed error extraction: now reads `result.error.traceback`

**Evidence**:
- Job `e749a6e9` (reddit): 3/4 crashed with `AttributeError: 'dict' object has no attribute 'id'` at verifier.py line 37

---

## Fix #6: Base Quality Reward for GRPO Signal
**Commit**: (pending)
**Symptom**: Rewards stuck at 0.0 despite working accumulators. Verifier structure correct but logic wrong (bad column names, wrong table lookups). All harness evals return 0 → GRPO has zero variance → no learning signal.
**Root Cause**: When ALL harness evaluations score 0.0, `compute_task_reward()` returns `var=0, hint_gap=0, total=0`. With all samples getting the same reward, GRPO advantage is zero for every token.
**Fix**:
- Added `base_quality_reward` parameter (default 0.1) to TaskGenEnv
- Tasks passing sandbox+judge gate get `R = 0.1 + eval_signal` instead of `R = eval_signal`
- Tasks failing parse/sandbox/judge still get R=0.0
- This creates reward variance between "structurally valid" (0.1) and "invalid" (0.0) samples, giving GRPO gradient signal to push toward valid task generation
**Expected reward distribution**:
- Parse failure → 0.0
- Sandbox failure → 0.0
- Judge failure → 0.0
- Pass sandbox+judge, harness all zeros → 0.1
- Pass sandbox+judge, some harness success → 0.1 + eval_signal (up to ~1.1)

**Evidence** (iter4, first 2 steps):
- Step 1: avg_final_rewards=0.0238 (was 0.0 in all prior runs!)
- Step 2: avg_final_rewards=0.0292
- taskgen_1c5278de: raw=[0,1,0,0] hinted=[1,1,1,1] total=0.8438
- taskgen_f6c1299c: raw=[0,0,0,0] hinted=[1,1,1,1] total=1.0000

---

## Fix #7: Address 5 Verifier Crash Modes
**Commit**: (pending)
**Symptom**: 89% of harness evals still score 0. Investigation of 160 sessions across 40 jobs revealed 5 distinct verifier crash categories.
**Root Causes** (from iter4 job analysis):
1. **Hallucinated `.order()` method** (7 crashes): Model calls `.order("col", descending=True)` which doesn't exist
2. **`.eq()` with 3 args** (5 crashes): Model writes `.eq("rating", ">", 8.0)` — `.eq()` only takes `(column, value)`
3. **Tuple vs dict confusion** (5 crashes): Creates tuples in list comp `[(a, b) for ...]`, then tries `item["key"]` access
4. **KeyError after `.select()`** (8 crashes): Uses `.select("id", "name")` then accesses `row["subscribers"]`
5. **`find_new_entries` not defined** (5 crashes): Calls helper without defining it in verifier body
6. **Hardcoded expected values** (24+ sessions): Verifier checks for specific invented values agent can't know
**Fix**: Added rules to system prompt:
- Explicitly banned `.order()`, `.limit()` — use Python `sorted()`/`[:N]` instead
- Documented `.eq()` takes exactly 2 args, use Python for comparisons
- Warned about tuple/dict confusion in list comprehensions
- Warned that `.select()` limits available columns
- Instructed to define `find_new_entries` inside verifier body (not a built-in)
- Added rule: never hardcode expected values agent must invent — compare against seed state instead

---

## Known Issues (Not Yet Fixed)

### Fleet Harness Runtime Bug: `env.env_variables = None`
The Fleet harness verifier runtime (`/app/fleetgen/runtime.py`) creates an `Environment` object but doesn't populate `env_variables` from the task's stored env_variables. This is a server-side bug — workaround is to not use this API in generated verifiers.

### Bug #13: Evaluator Agent Asks Follow-up Questions
The evaluator agent (Sonnet 4.5) sometimes asks follow-up questions instead of completing the task. Lower priority — deprioritized by user.

---

## Training Runs

| Run Name | WandB | Status | Notes |
|----------|-------|--------|-------|
| `task_gen_494dce32` | `9i3ueeut` | Killed | Fix #1-#2 only, 100% zero rewards |
| `task_gen_55f7b9c8` | TBD | Killed | Fix #1-#3, still all zeros (env.env_variables crash) |
| iter3 (55f7b9c8 cont) | TBD | Killed | Fix #1-#4, still zeros (dict access crash) |
| iter3 (relaunched) | TBD | Killed | Fix #1-#5, accumulators work but all harness 0.0 |
| iter4 (task_gen_c1e71be3) | `1hsk4bhw` | Running | Fix #1-#6, base_quality=0.1, first non-zero rewards! |
