from __future__ import annotations

from typing import Dict, Optional

import numpy as np

import pyspiel
from open_spiel.python.algorithms import cfr


class CFRPolicy:
    """CFR policy trained on the full game and exposed as a callable opponent."""

    def __init__(
        self,
        game_name: str,
        iterations: int = 1000,
        algorithm: str = "cfr",
        seed: Optional[int] = None,
    ) -> None:
        self._game = pyspiel.load_game(game_name)
        self._rng = np.random.default_rng(seed)
        self._policy = self._train(iterations, algorithm)

    def _train(self, iterations: int, algorithm: str):
        if algorithm == "cfr":
            solver = cfr.CFRSolver(self._game)
        elif algorithm == "cfr_plus":
            solver = cfr.CFRPlusSolver(self._game)
        else:
            raise ValueError(f"Unsupported CFR algorithm: {algorithm}")

        for _ in range(iterations):
            solver.evaluate_and_update_policy()
        return solver.average_policy()

    def action_probabilities(self, state: pyspiel.State) -> Dict[int, float]:
        return self._policy.action_probabilities(state)

    def action(self, state: pyspiel.State) -> int:
        probs = self.action_probabilities(state)
        actions, weights = zip(*probs.items())
        return int(self._rng.choice(actions, p=np.array(weights)))

    def __call__(self, state: pyspiel.State, rng: np.random.Generator) -> int:
        probs = self.action_probabilities(state)
        actions, weights = zip(*probs.items())
        return int(rng.choice(actions, p=np.array(weights)))
