"""Numerically stable scalar and torch math for the UltraFeedback TRO study.

The functions in this module implement the protocol's exact binary KL ball
and per-pair TRO objective.  Targets and interval endpoints are data-derived
and are deliberately detached from the model graph in the torch helper.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def bernoulli_kl(empirical: float, candidate: float) -> float:
    if not (0.0 <= empirical <= 1.0 and 0.0 <= candidate <= 1.0):
        raise ValueError("Bernoulli probabilities must be in [0, 1]")
    if empirical == candidate:
        return 0.0
    if candidate == 0.0 or candidate == 1.0:
        return math.inf
    if empirical == 0.0:
        return -math.log1p(-candidate)
    if empirical == 1.0:
        return -math.log(candidate)
    return empirical * (math.log(empirical) - math.log(candidate)) + (1.0 - empirical) * (
        math.log1p(-empirical) - math.log1p(-candidate)
    )


def kl_interval(empirical: float, radius: float, steps: int = 100) -> Tuple[float, float]:
    """Return [lower, upper] for KL(Ber(empirical)||Ber(p)) <= radius."""
    if not (0.0 <= empirical <= 1.0 and math.isfinite(radius) and radius >= 0.0):
        raise ValueError("invalid empirical probability or KL radius")
    if radius == 0.0:
        return empirical, empirical
    if empirical == 0.0:
        return 0.0, -math.expm1(-radius)
    if empirical == 1.0:
        return math.exp(-radius), 1.0

    outside, inside = 0.0, empirical
    for _ in range(steps):
        mid = (outside + inside) / 2.0
        if bernoulli_kl(empirical, mid) > radius:
            outside = mid
        else:
            inside = mid
    lower = inside

    inside, outside = empirical, 1.0
    for _ in range(steps):
        mid = (inside + outside) / 2.0
        if bernoulli_kl(empirical, mid) > radius:
            outside = mid
        else:
            inside = mid
    return lower, inside


def n_radius(n_votes: int, delta: float = 0.05, eta: float = 1.0) -> float:
    if n_votes <= 0 or not (0.0 < delta < 1.0) or eta < 0.0:
        raise ValueError("invalid n-radius arguments")
    return eta * math.log((n_votes + 1.0) / delta) / n_votes


def lambda_from_n(n_votes: int, n0: float) -> float:
    """Return v5 trust weight N/(N+c).

    The v3 implementation used the same symbol for the robust-component
    weight c/(N+c).  v5 names this quantity lambda_trust, so it weights the
    empirical soft loss directly and the worst-case loss by (1-lambda).
    """
    if n_votes <= 0 or n0 < 0.0:
        raise ValueError("invalid lambda arguments")
    return 0.0 if math.isinf(n0) else n_votes / (n_votes + n0)


def tro_loss_from_margin(z: float, p_hat: float, lower: float, upper: float, lam: float):
    """Return scalar TRO loss and d(loss)/d(z), using the protocol zero rule."""
    if not (0.0 <= lower <= p_hat <= upper <= 1.0 and 0.0 <= lam <= 1.0):
        raise ValueError("invalid TRO interval")
    if z > 0.0:
        worst = lower
    elif z < 0.0:
        worst = upper
    else:
        worst = p_hat
    # v5 semantics: lambda is trust in the empirical proportion.
    target = lam * p_hat + (1.0 - lam) * worst
    loss = target * F.softplus(torch.tensor(-z, dtype=torch.float64)).item() + (1.0 - target) * F.softplus(
        torch.tensor(z, dtype=torch.float64)
    ).item()
    # sigmoid(z)-target is the exact derivative for fixed target on each side.
    sig = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
    return loss, sig - target


def binary_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    target: torch.Tensor,
):
    """Per-pair cross entropy for a fixed binary target in response-1 order."""
    z = beta * (
        (policy_chosen_logps.float() - reference_chosen_logps.float())
        - (policy_rejected_logps.float() - reference_rejected_logps.float())
    )
    target = target.float().detach().clamp(0.0, 1.0)
    per_pair = -target * F.logsigmoid(z) - (1.0 - target) * F.logsigmoid(-z)
    return per_pair, z


def tro_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    p_hat: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lam: torch.Tensor,
):
    """Return (per-pair loss, margin, effective target, worst endpoint)."""
    per_soft, z = binary_dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
        p_hat,
    )
    p_hat = p_hat.float().detach()
    lower = lower.float().detach()
    upper = upper.float().detach()
    lam = lam.float().detach().clamp(0.0, 1.0)
    worst = torch.where(z > 0.0, lower, torch.where(z < 0.0, upper, p_hat))
    # v5 semantics: lambda is trust in the empirical proportion.
    target = lam * p_hat + (1.0 - lam) * worst
    per_pair = -target * F.logsigmoid(z) - (1.0 - target) * F.logsigmoid(-z)
    return per_pair, z, target, worst


def dro_target(z: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, p_hat: torch.Tensor):
    """Worst-case probability for label DRO; the zero margin uses center."""
    return torch.where(z > 0.0, lower, torch.where(z < 0.0, upper, p_hat)).detach()
