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
| (next) | TBD | Pending | Fix #1-#4, env vars as constants + error hints |
