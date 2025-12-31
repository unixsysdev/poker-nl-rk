from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


try:
    from treys import Card, Deck, Evaluator
except ImportError as exc:  # pragma: no cover
    raise ImportError("treys is required. Install it with: pip install treys") from exc


@dataclass
class EnvConfig:
    num_players: int = 2
    stack: int = 20000
    small_blind: int = 50
    big_blind: int = 100
    max_raises_per_round: int = 4
    ante: int = 0
    rake_pct: float = 0.0
    rake_cap: int = 0
    rake_cap_per_hand: int = 0
    rake_cap_per_street: int = 0
    history_len: int = 12
    hands_per_episode: int = 1
    seed: int = 42


class NoLimitHoldemEnv:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.evaluator = Evaluator()
        self.full_deck = Deck.GetFullDeck()
        self.card_index = {card: idx for idx, card in enumerate(self.full_deck)}
        self.obs_dim = 52 + 5 + 7 * self.config.num_players + 4 * self.config.history_len
        self.reset()

    def reset(self):
        self.hands_left = self.config.hands_per_episode
        self.stacks = [self.config.stack for _ in range(self.config.num_players)]
        self.starting_stacks = self.stacks[:]
        self.busted = [False for _ in range(self.config.num_players)]
        self.dealer = int(self.rng.integers(0, self.config.num_players))
        self.episode_over = False
        self._start_hand(new_hand=False)
        return self.get_state(self.current_player), self.current_player

    def _start_hand(self, new_hand: bool):
        self.deck = Deck()
        self.board = []
        self.hole_cards = []
        for i in range(self.config.num_players):
            if self.busted[i]:
                self.hole_cards.append([])
            else:
                self.hole_cards.append([self.deck.draw(1)[0], self.deck.draw(1)[0]])
        self.in_pot = [0 for _ in range(self.config.num_players)]
        self.street_bet = [0 for _ in range(self.config.num_players)]
        self.last_action = [-1 for _ in range(self.config.num_players)]
        self.last_bet_frac = [0.0 for _ in range(self.config.num_players)]
        self.folded = [self.busted[i] for i in range(self.config.num_players)]
        self.all_in = [False for _ in range(self.config.num_players)]
        self.action_history = []
        self.round_index = 0
        self.raise_count = 0
        self.rake_taken_hand = 0
        self.rake_taken_street = [0, 0, 0, 0]

        if new_hand:
            self.dealer = self._next_eligible(self.dealer)

        self.current_player = self._next_eligible(self.dealer)

        if self.config.ante > 0:
            for i in self._eligible_players():
                self._bet(i, self.config.ante)
        self._post_blinds()
        self.pending = self._actionable_players(exclude=self.last_aggressor)

    def _next_eligible(self, start):
        idx = start
        for _ in range(self.config.num_players):
            idx = (idx + 1) % self.config.num_players
            if not self.busted[idx]:
                return idx
        return start

    def _post_blinds(self):
        if self.config.num_players == 2:
            sb_player = self.dealer
            bb_player = self._next_player(sb_player)
        else:
            sb_player = self._next_player(self.dealer)
            bb_player = self._next_player(sb_player)
        self._bet(sb_player, self.config.small_blind)
        self._bet(bb_player, self.config.big_blind)
        self.street_bet[sb_player] = self.config.small_blind
        self.street_bet[bb_player] = self.config.big_blind
        self.current_bet = self.config.big_blind
        self.min_raise = self.config.big_blind
        self.last_aggressor = bb_player
        self.current_player = self._next_player(bb_player)

    def _next_player(self, start):
        idx = start
        for _ in range(self.config.num_players):
            idx = (idx + 1) % self.config.num_players
            if not self.folded[idx] and not self.all_in[idx] and not self.busted[idx]:
                return idx
        return start

    def _actionable_players(self, exclude: Optional[int] = None) -> List[int]:
        players = [
            i
            for i in range(self.config.num_players)
            if not self.folded[i] and not self.all_in[i] and not self.busted[i]
        ]
        if exclude is not None and exclude in players:
            players.remove(exclude)
        return players

    def _eligible_players(self) -> List[int]:
        return [i for i in range(self.config.num_players) if not self.folded[i] and not self.busted[i]]

    def _bet(self, player: int, amount: int):
        amount = min(amount, self.stacks[player])
        self.stacks[player] -= amount
        self.in_pot[player] += amount
        if self.stacks[player] == 0 and not self.folded[player]:
            self.all_in[player] = True
        return amount

    def _advance_round(self):
        self.raise_count = 0
        self.current_bet = 0
        self.min_raise = self.config.big_blind
        self.last_aggressor = None
        self.pending = self._actionable_players()
        self.street_bet = [0 for _ in range(self.config.num_players)]

        if self.round_index == 0:
            self.board += self.deck.draw(3)
        elif self.round_index == 1:
            self.board += self.deck.draw(1)
        elif self.round_index == 2:
            self.board += self.deck.draw(1)
        else:
            return

        self.round_index += 1
        if self.pending:
            self.current_player = self._next_player(self.dealer)
        else:
            self.current_player = self._next_eligible(self.dealer)

    def _is_terminal(self):
        if self.episode_over:
            return True
        return False

    def _resolve_showdown(self):
        active = self._eligible_players()
        if not active:
            return

        contributions = self.in_pot[:]
        levels = sorted({c for c in contributions if c > 0})
        prev = 0
        for level in levels:
            pot = (level - prev) * sum(1 for c in contributions if c >= level)
            pot = self._apply_rake(pot, street_index=3)
            eligible = [i for i in active if contributions[i] >= level]
            if not eligible:
                prev = level
                continue
            scores = [(self.evaluator.evaluate(self.board, self.hole_cards[i]), i) for i in eligible]
            best = min(scores, key=lambda x: x[0])[0]
            winners = [i for score, i in scores if score == best]
            share = pot // len(winners)
            remainder = pot - share * len(winners)
            for i in winners:
                self.stacks[i] += share
            if remainder > 0:
                winners_sorted = sorted(winners, key=lambda i: (i - self.dealer) % self.config.num_players)
                self.stacks[winners_sorted[0]] += remainder
            prev = level

    def _resolve_fold(self):
        active = self._eligible_players()
        if len(active) == 1:
            winner = active[0]
            pot = self._apply_rake(sum(self.in_pot), street_index=min(self.round_index, 3))
            self.stacks[winner] += pot

    def _apply_rake(self, pot: int, street_index: Optional[int] = None) -> int:
        if self.config.rake_pct <= 0:
            return pot
        rake = int(pot * self.config.rake_pct)
        if self.config.rake_cap > 0:
            rake = min(rake, self.config.rake_cap)
        if self.config.rake_cap_per_hand > 0:
            rake = min(rake, max(0, self.config.rake_cap_per_hand - self.rake_taken_hand))
        if street_index is not None and self.config.rake_cap_per_street > 0:
            rake = min(
                rake,
                max(0, self.config.rake_cap_per_street - self.rake_taken_street[street_index]),
            )
        rake = max(0, rake)
        self.rake_taken_hand += rake
        if street_index is not None:
            self.rake_taken_street[street_index] += rake
        return max(0, pot - rake)

    def _finish_hand(self):
        self.hands_left -= 1
        for i in range(self.config.num_players):
            if self.stacks[i] <= 0:
                self.busted[i] = True
        eligible = [i for i in range(self.config.num_players) if not self.busted[i]]
        if self.hands_left <= 0 or len(eligible) <= 1:
            self.episode_over = True
            return
        self._start_hand(new_hand=True)

    def _record_action(self, player: int, action_type: int, bet_frac: float):
        self.action_history.append((player, action_type, bet_frac, self.round_index))

    def get_payoffs(self) -> List[float]:
        return [stack - baseline for stack, baseline in zip(self.stacks, self.starting_stacks)]

    def legal_action_mask(self, player: int) -> np.ndarray:
        mask = np.zeros(3, dtype=np.float32)  # fold, call/check, raise
        if self.folded[player] or self.all_in[player] or self.busted[player]:
            return mask
        to_call = max(0, self.current_bet - self.street_bet[player])
        if to_call > 0:
            mask[0] = 1.0
        else:
            mask[0] = 0.0
        mask[1] = 1.0
        can_raise = (
            self.raise_count < self.config.max_raises_per_round
            and self.stacks[player] > to_call
        )
        if can_raise:
            mask[2] = 1.0
        return mask

    def get_state(self, player: int) -> Dict:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        for card in self.hole_cards[player]:
            obs[self.card_index[card]] = 1.0
        for card in self.board:
            obs[self.card_index[card]] = 1.0

        offset = 52
        obs[offset] = float(sum(self.in_pot)) / self.config.big_blind
        obs[offset + 1] = float(self.current_bet) / self.config.big_blind
        obs[offset + 2] = float(max(0, self.current_bet - self.street_bet[player])) / self.config.big_blind
        obs[offset + 3] = float(self.round_index)
        obs[offset + 4] = float((player - self.dealer) % self.config.num_players) / self.config.num_players

        offset += 5
        for i in range(self.config.num_players):
            obs[offset + i] = float(self.stacks[i]) / self.config.big_blind
        offset += self.config.num_players
        for i in range(self.config.num_players):
            obs[offset + i] = float(self.in_pot[i]) / self.config.big_blind
        offset += self.config.num_players
        for i in range(self.config.num_players):
            obs[offset + i] = float(self.street_bet[i]) / self.config.big_blind
        offset += self.config.num_players
        for i in range(self.config.num_players):
            obs[offset + i] = 1.0 if self.folded[i] else 0.0
        offset += self.config.num_players
        for i in range(self.config.num_players):
            obs[offset + i] = 1.0 if self.all_in[i] else 0.0
        offset += self.config.num_players
        for i in range(self.config.num_players):
            obs[offset + i] = float(self.last_action[i] + 1) / 3.0
        offset += self.config.num_players
        for i in range(self.config.num_players):
            obs[offset + i] = float(self.last_bet_frac[i])
        offset += self.config.num_players

        history = self.action_history[-self.config.history_len :]
        for idx, (player, action_type, bet_frac, round_index) in enumerate(history):
            base = offset + idx * 4
            obs[base] = float(action_type + 1) / 3.0
            obs[base + 1] = float(bet_frac)
            obs[base + 2] = float((player - self.dealer) % self.config.num_players) / self.config.num_players
            obs[base + 3] = float(round_index) / 3.0

        return {
            "obs": obs,
            "legal_action_mask": self.legal_action_mask(player),
            "player_id": player,
            "round": self.round_index,
            "pot": sum(self.in_pot),
            "current_bet": self.current_bet,
            "to_call": max(0, self.current_bet - self.street_bet[player]),
        }

    def step(self, action_type: int, bet_fraction: float):
        if self._is_terminal():
            return self.get_state(self.current_player), self.current_player

        if not self._actionable_players():
            while self.round_index < 3:
                self._advance_round()
            self.round_index = 4
            self._resolve_showdown()
            self._finish_hand()
            return self.get_state(self.current_player), self.current_player

        player = self.current_player
        if self.folded[player] or self.all_in[player] or self.busted[player]:
            self.current_player = self._next_player(player)
            return self.get_state(self.current_player), self.current_player

        bet_fraction = float(np.clip(bet_fraction, 0.0, 1.0))
        scaled_frac = bet_fraction * 2.0
        to_call = max(0, self.current_bet - self.street_bet[player])
        legal = self.legal_action_mask(player)

        actual_action = 1
        actual_bet_frac = 0.0
        if action_type == 0 and legal[0] == 1.0:
            self.folded[player] = True
            self.pending = [p for p in self.pending if p != player]
            actual_action = 0
        elif action_type == 2 and legal[2] == 1.0:
            pot = max(1, sum(self.in_pot) + to_call)
            raise_amount = max(self.min_raise, int(scaled_frac * pot))
            total = to_call + raise_amount
            paid = self._bet(player, total)
            self.street_bet[player] += paid
            actual_action = 2
            actual_bet_frac = scaled_frac
            if paid > to_call:
                actual_raise = paid - to_call
                if self.street_bet[player] > self.current_bet:
                    self.current_bet = self.street_bet[player]
                if actual_raise >= self.min_raise:
                    self.min_raise = max(self.min_raise, actual_raise)
                    self.raise_count += 1
                    self.last_aggressor = player
                    self.pending = self._actionable_players(exclude=player)
                else:
                    self.pending = [p for p in self.pending if p != player]
            else:
                self.pending = [p for p in self.pending if p != player]
        else:
            paid = self._bet(player, to_call)
            self.street_bet[player] += paid
            self.pending = [p for p in self.pending if p != player]

        self.last_action[player] = actual_action
        self.last_bet_frac[player] = actual_bet_frac
        self._record_action(player, actual_action, actual_bet_frac)

        if len(self._eligible_players()) <= 1:
            self._resolve_fold()
            self.round_index = 4
            self._finish_hand()
            return self.get_state(self.current_player), self.current_player

        if not self._actionable_players():
            while self.round_index < 3:
                self._advance_round()
            self.round_index = 4
            self._resolve_showdown()
            self._finish_hand()
            return self.get_state(self.current_player), self.current_player

        if not self.pending:
            if self.round_index < 3:
                self._advance_round()
            else:
                self.round_index = 4
                self._resolve_showdown()
                self._finish_hand()
                return self.get_state(self.current_player), self.current_player

        self.current_player = self._next_player(self.current_player)
        return self.get_state(self.current_player), self.current_player

    def is_over(self):
        return self._is_terminal()
