from __future__ import annotations

from typing import Dict


def limit_holdem_params(
    num_ranks: int = 6,
    num_suits: int = 4,
    stack: str = "2000 2000",
    blind: str = "50 100",
    raise_size: str = "100 100 200 200",
    max_raises: str = "4 4 4 4",
    first_player: str = "1 1 1 1",
) -> Dict[str, str]:
    """Approximate 2-player Limit Hold'em using universal_poker parameters.

    Uses a reduced deck by default (num_ranks=6) to keep training tractable.
    """
    return {
        "betting": "limit",
        "bettingAbstraction": "fcpa",
        "blind": blind,
        "boardCards": "",
        "firstPlayer": first_player,
        "maxRaises": max_raises,
        "numBoardCards": "0 3 1 1",
        "numHoleCards": 2,
        "numPlayers": 2,
        "numRanks": num_ranks,
        "numRounds": 4,
        "numSuits": num_suits,
        "potSize": 0,
        "raiseSize": raise_size,
        "stack": stack,
    }


def nlhe_params(
    num_ranks: int = 13,
    num_suits: int = 4,
    stack: str = "2000 2000",
    blind: str = "50 100",
    betting_abstraction: str = "fcpa",
    raise_size: str = "100 100 100 100",
    max_raises: str = "4 4 4 4",
    first_player: str = "1 1 1 1",
) -> Dict[str, str]:
    """No-Limit Hold'em-like config with an action abstraction."""
    return {
        "betting": "nolimit",
        "bettingAbstraction": betting_abstraction,
        "blind": blind,
        "boardCards": "",
        "firstPlayer": first_player,
        "maxRaises": max_raises,
        "numBoardCards": "0 3 1 1",
        "numHoleCards": 2,
        "numPlayers": 2,
        "numRanks": num_ranks,
        "numRounds": 4,
        "numSuits": num_suits,
        "potSize": 0,
        "raiseSize": raise_size,
        "stack": stack,
    }
