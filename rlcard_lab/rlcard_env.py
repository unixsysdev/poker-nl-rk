from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


try:
    import rlcard
    from rlcard.games.nolimitholdem.round import Action
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "rlcard is not installed. Install it with: pip install rlcard"
    ) from exc


NUM_ACTIONS = 5
OBS_SHAPE = 54


ACTION_NAMES = {
    Action.FOLD.value: "fold",
    Action.CHECK_CALL.value: "check/call",
    Action.RAISE_HALF_POT.value: "raise_half_pot",
    Action.RAISE_POT.value: "raise_pot",
    Action.ALL_IN.value: "all_in",
}


@dataclass(frozen=True)
class ActionMapping:
    fold: int = Action.FOLD.value
    call: int = Action.CHECK_CALL.value
    raise_half_pot: int = Action.RAISE_HALF_POT.value
    raise_pot: int = Action.RAISE_POT.value
    all_in: int = Action.ALL_IN.value


def make_env(seed: int, num_players: int = 2, allow_step_back: bool = False):
    config = {
        "game_num_players": num_players,
        "seed": seed,
        "allow_step_back": allow_step_back,
    }
    env = rlcard.make("no-limit-holdem", config=config)
    return env


def legal_action_mask(state: Dict) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for action_id in state["legal_actions"].keys():
        mask[action_id] = 1.0
    return mask


def describe_legal_actions(state: Dict) -> List[str]:
    return [ACTION_NAMES[a] for a in state["legal_actions"].keys()]
