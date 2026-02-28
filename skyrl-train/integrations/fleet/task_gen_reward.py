"""
Reward functions for task generation RL.

Computes:
    R(task) = validity * (variance + alpha * separation)

Components:
    - Learnability (variance): Mean variance of solve outcomes across k rollouts per model.
      Maximized when p_solve ≈ 0.5 (max Bernoulli variance = 0.25).
    - Separation: Gap between strongest and weakest model solve rates.
      Higher = task better differentiates model capabilities.
    - Validity: Multiplicative gate from verifier_sandbox.py.
      Returns 0 for broken tasks, killing the entire reward.
"""

from typing import Any, Dict, List


def compute_learnability(results_per_model: Dict[str, List[float]]) -> float:
    """Compute learnability as mean rollout variance, normalized to [0, 1].

    For each model, compute the variance of k binary rollout outcomes.
    Average across models and normalize by max Bernoulli variance (0.25).

    Highest signal when p_solve ≈ 0.5 (some rollouts pass, some fail).
    Zero signal when p_solve ≈ 0 or p_solve ≈ 1 (all same outcome).

    Args:
        results_per_model: {model_id: [reward_1, reward_2, ..., reward_k]}

    Returns:
        Normalized learnability score in [0, 1].
    """
    if not results_per_model:
        return 0.0

    variances = []
    for results in results_per_model.values():
        if len(results) < 2:
            variances.append(0.0)
            continue
        mean = sum(results) / len(results)
        var = sum((r - mean) ** 2 for r in results) / len(results)
        variances.append(var)

    mean_var = sum(variances) / len(variances)
    # Normalize by max Bernoulli variance (0.25) to get [0, 1]
    return min(mean_var / 0.25, 1.0)


def compute_separation(results_per_model: Dict[str, List[float]]) -> float:
    """Compute model separation as solve rate gap.

    The gap between the best and worst model's solve rate.
    Higher = task better differentiates model capabilities.

    Args:
        results_per_model: {model_id: [reward_1, reward_2, ..., reward_k]}

    Returns:
        Separation score in [0, 1].
    """
    if len(results_per_model) < 2:
        return 0.0

    solve_rates = {}
    for model_id, results in results_per_model.items():
        if results:
            solve_rates[model_id] = sum(results) / len(results)
        else:
            solve_rates[model_id] = 0.0

    return max(solve_rates.values()) - min(solve_rates.values())


def compute_composite_reward(
    results_per_model: Dict[str, List[float]],
    validity: float = 1.0,
    alpha: float = 0.5,
) -> Dict[str, float]:
    """Compute the full composite reward.

    R(task) = validity * (learnability + alpha * separation)

    Args:
        results_per_model: {model_id: [reward_1, ..., reward_k]}
        validity: Multiplicative gate (0.0 or 1.0)
        alpha: Weight for separation term

    Returns:
        Dict with all reward components and total.
    """
    learnability = compute_learnability(results_per_model)
    separation = compute_separation(results_per_model)
    total = validity * (learnability + alpha * separation)

    return {
        "validity": validity,
        "learnability": learnability,
        "separation": separation,
        "alpha": alpha,
        "total": total,
    }


def compute_per_model_stats(
    results_per_model: Dict[str, List[float]],
) -> Dict[str, Dict[str, Any]]:
    """Compute per-model statistics for logging.

    Args:
        results_per_model: {model_id: [reward_1, ..., reward_k]}

    Returns:
        {model_id: {solve_rate, variance, k}} for each model.
    """
    stats = {}
    for model_id, results in results_per_model.items():
        if not results:
            stats[model_id] = {"solve_rate": 0.0, "variance": 0.0, "k": 0}
            continue

        solve_rate = sum(results) / len(results)
        mean = solve_rate
        var = sum((r - mean) ** 2 for r in results) / len(results) if len(results) > 1 else 0.0

        stats[model_id] = {
            "solve_rate": solve_rate,
            "variance": var,
            "k": len(results),
        }

    return stats
