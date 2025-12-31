from __future__ import annotations

from typing import Callable, Optional

import numpy as np

import pyspiel

from poker_rl.solvers.cfr_policy import CFRPolicy


PolicyCallable = Callable[[pyspiel.State, np.random.Generator], int]


class SolverAuditPolicy:
    """Audit a candidate action against a CFR policy and optionally override."""

    def __init__(
        self,
        base_policy: Optional[PolicyCallable] = None,
        solver_policy: Optional[CFRPolicy] = None,
        min_solver_prob: float = 0.05,
        fallback: str = "argmax",
        seed: Optional[int] = None,
    ) -> None:
        if fallback not in {"argmax", "sample"}:
            raise ValueError("fallback must be 'argmax' or 'sample'.")
        self._base_policy = base_policy
        self._solver_policy = solver_policy
        self._min_solver_prob = min_solver_prob
        self._fallback = fallback
        self._rng = np.random.default_rng(seed)

    def set_base_policy(self, base_policy: PolicyCallable) -> None:
        self._base_policy = base_policy

    def set_solver_policy(self, solver_policy: CFRPolicy) -> None:
        self._solver_policy = solver_policy

    def action(self, state: pyspiel.State) -> int:
        if self._base_policy is None:
            raise RuntimeError("Base policy is not set.")
        candidate = self._base_policy(state, self._rng)
        return self.audit_action(state, candidate)

    def audit_action(self, state: pyspiel.State, candidate_action: int) -> int:
        if self._solver_policy is None:
            return candidate_action

        legal_actions = state.legal_actions(state.current_player())
        if candidate_action not in legal_actions:
            raise ValueError(
                f"Illegal action {candidate_action}; legal: {legal_actions}"
            )

        probs = self._solver_policy.action_probabilities(state)
        candidate_prob = probs.get(candidate_action, 0.0)
        if candidate_prob >= self._min_solver_prob:
            return candidate_action

        if self._fallback == "argmax":
            return max(probs, key=probs.get)
        actions, weights = zip(*probs.items())
        return int(self._rng.choice(actions, p=np.array(weights)))
