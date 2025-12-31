from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
import numpy as np

try:
    import pyspiel
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "OpenSpiel is required. Install with `pip install open_spiel`."
    ) from exc


OpponentPolicy = Callable[[pyspiel.State, np.random.Generator], int]


def _random_opponent(state: pyspiel.State, rng: np.random.Generator) -> int:
    legal_actions = state.legal_actions(state.current_player())
    if not legal_actions:
        raise RuntimeError("No legal actions available for opponent.")
    return int(rng.choice(legal_actions))


class OpenSpielPokerEnv(gym.Env):
    """Gymnasium wrapper around OpenSpiel poker games (Leduc, Hold'em, etc.).

    This is a single-agent view with an internal opponent policy. It is meant for
    fast iteration and baseline training, not as a full multi-agent interface.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        game_name: str = "leduc_poker",
        player_id: int = 0,
        obs_type: str = "info_state",
        opponent_policy: Optional[OpponentPolicy] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._game = pyspiel.load_game(game_name)
        self._player_id = player_id
        self._obs_type = obs_type
        self._opponent_policy = opponent_policy or _random_opponent
        self._rng = np.random.default_rng(seed)

        if obs_type == "info_state":
            obs_dim = self._game.information_state_tensor_size()
        elif obs_type == "observation":
            obs_dim = self._game.observation_tensor_size()
        else:
            raise ValueError(f"Unsupported obs_type: {obs_type}")

        self._num_actions = self._game.num_distinct_actions()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(self._num_actions)
        self._state = self._game.new_initial_state()

    @property
    def state(self) -> pyspiel.State:
        return self._state

    def set_opponent_policy(self, opponent_policy: OpponentPolicy) -> None:
        self._opponent_policy = opponent_policy

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._game.new_initial_state()
        self._advance_until_player()
        return self._get_obs(), self._info()

    def step(self, action: int):
        if self._state.is_terminal():
            raise RuntimeError("step() called on terminal state.")

        current_player = self._state.current_player()
        if current_player != self._player_id:
            raise RuntimeError(
                f"Expected player {self._player_id}, got {current_player}."
            )

        legal_actions = self._state.legal_actions(current_player)
        if action not in legal_actions:
            raise ValueError(f"Illegal action {action}; legal: {legal_actions}")

        self._state.apply_action(action)
        self._advance_until_player()

        terminated = self._state.is_terminal()
        reward = float(self._state.returns()[self._player_id]) if terminated else 0.0
        return self._get_obs(), reward, terminated, False, self._info()

    def _advance_until_player(self) -> None:
        while not self._state.is_terminal():
            current_player = self._state.current_player()
            if current_player == self._player_id:
                break

            if self._state.is_chance_node():
                action = self._sample_chance_action()
            else:
                action = self._opponent_policy(self._state, self._rng)

            self._state.apply_action(action)

    def _sample_chance_action(self) -> int:
        outcomes = self._state.chance_outcomes()
        if not outcomes:
            raise RuntimeError("Chance node has no outcomes.")
        actions, probs = zip(*outcomes)
        return int(self._rng.choice(actions, p=probs))

    def _get_obs(self) -> np.ndarray:
        if self._obs_type == "info_state":
            obs = self._state.information_state_tensor(self._player_id)
        else:
            obs = self._state.observation_tensor(self._player_id)
        return np.asarray(obs, dtype=np.float32)

    def _info(self) -> dict:
        if self._state.is_terminal():
            legal_actions = []
        else:
            legal_actions = self._state.legal_actions(self._player_id)
        mask = np.zeros(self._num_actions, dtype=np.int8)
        if legal_actions:
            mask[legal_actions] = 1
        return {
            "legal_actions": legal_actions,
            "legal_actions_mask": mask,
            "current_player": self._state.current_player(),
            "state": self._state,
        }
