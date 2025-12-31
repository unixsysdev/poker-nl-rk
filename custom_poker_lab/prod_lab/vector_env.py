from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    from treys import Deck, Evaluator
except ImportError as exc:  # pragma: no cover
    raise ImportError("treys is required. Install it with: pip install treys") from exc


@dataclass
class VectorEnvConfig:
    batch_size: int = 64
    num_players: int = 6
    stack: int = 20000
    small_blind: int = 50
    big_blind: int = 100
    max_raises_per_round: int = 0  # 0 = unlimited
    ante: int = 0
    rake_pct: float = 0.0
    rake_cap: int = 0
    rake_cap_per_hand: int = 0
    rake_cap_per_street: int = 0
    history_len: int = 12
    hands_per_episode: int = 1
    seed: int = 42


class VectorNLHEEnv:
    def __init__(self, config: VectorEnvConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.evaluator = Evaluator()
        self.full_deck = np.array(Deck.GetFullDeck(), dtype=np.int64)
        self.card_index = {int(card): idx for idx, card in enumerate(self.full_deck)}
        self.obs_dim = 52 + 5 + 7 * self.config.num_players + 4 * self.config.history_len
        self._allocate()
        self.reset()

    def _allocate(self):
        b = self.config.batch_size
        n = self.config.num_players
        self.stacks = np.zeros((b, n), dtype=np.int64)
        self.starting_stacks = np.zeros((b, n), dtype=np.int64)
        self.busted = np.zeros((b, n), dtype=bool)
        self.folded = np.zeros((b, n), dtype=bool)
        self.all_in = np.zeros((b, n), dtype=bool)
        self.in_pot = np.zeros((b, n), dtype=np.int64)
        self.street_bet = np.zeros((b, n), dtype=np.int64)
        self.last_action = np.full((b, n), -1, dtype=np.int64)
        self.last_bet_frac = np.zeros((b, n), dtype=np.float32)
        self.current_bet = np.zeros(b, dtype=np.int64)
        self.min_raise = np.zeros(b, dtype=np.int64)
        self.raise_count = np.zeros(b, dtype=np.int64)
        self.last_aggressor = np.full(b, -1, dtype=np.int64)
        self.dealer = np.zeros(b, dtype=np.int64)
        self.current_player = np.zeros(b, dtype=np.int64)
        self.round_index = np.zeros(b, dtype=np.int64)
        self.hands_left = np.zeros(b, dtype=np.int64)
        self.episode_over = np.zeros(b, dtype=bool)
        self.pending = np.zeros((b, n), dtype=bool)
        self.deck = np.zeros((b, 52), dtype=np.int64)
        self.deck_pos = np.zeros(b, dtype=np.int64)
        self.board = np.full((b, 5), -1, dtype=np.int64)
        self.hole_cards = np.full((b, n, 2), -1, dtype=np.int64)
        self.rake_taken_hand = np.zeros(b, dtype=np.int64)
        self.rake_taken_street = np.zeros((b, 4), dtype=np.int64)
        self.hist_player = np.full((b, self.config.history_len), -1, dtype=np.int64)
        self.hist_action = np.full((b, self.config.history_len), -1, dtype=np.int64)
        self.hist_bet_frac = np.zeros((b, self.config.history_len), dtype=np.float32)
        self.hist_round = np.full((b, self.config.history_len), -1, dtype=np.int64)

    def reset(self):
        b = self.config.batch_size
        self.stacks[:] = self.config.stack
        self.starting_stacks[:] = self.stacks
        self.busted[:] = False
        self.hands_left[:] = self.config.hands_per_episode
        self.episode_over[:] = False
        for env in range(b):
            self.dealer[env] = int(self.rng.integers(0, self.config.num_players))
            self._start_hand(env, new_hand=False)
        return self.get_obs()

    def reset_at(self, env: int):
        self.stacks[env, :] = self.config.stack
        self.starting_stacks[env, :] = self.stacks[env, :]
        self.busted[env, :] = False
        self.hands_left[env] = self.config.hands_per_episode
        self.episode_over[env] = False
        self.dealer[env] = int(self.rng.integers(0, self.config.num_players))
        self._start_hand(env, new_hand=False)

    def _shuffle_deck(self, env: int):
        perm = self.rng.permutation(52)
        self.deck[env, :] = self.full_deck[perm]
        self.deck_pos[env] = 0

    def _draw(self, env: int, count: int) -> np.ndarray:
        start = self.deck_pos[env]
        end = start + count
        self.deck_pos[env] = end
        return self.deck[env, start:end]

    def _start_hand(self, env: int, new_hand: bool):
        self._shuffle_deck(env)
        self.board[env, :] = -1
        self.hole_cards[env, :, :] = -1
        self.in_pot[env, :] = 0
        self.street_bet[env, :] = 0
        self.last_action[env, :] = -1
        self.last_bet_frac[env, :] = 0.0
        self.folded[env, :] = self.busted[env, :]
        self.all_in[env, :] = False
        self.pending[env, :] = False
        self.round_index[env] = 0
        self.raise_count[env] = 0
        self.rake_taken_hand[env] = 0
        self.rake_taken_street[env, :] = 0
        self.hist_player[env, :] = -1
        self.hist_action[env, :] = -1
        self.hist_bet_frac[env, :] = 0.0
        self.hist_round[env, :] = -1

        if new_hand:
            self.dealer[env] = self._next_eligible(env, self.dealer[env])

        for player in range(self.config.num_players):
            if not self.busted[env, player]:
                cards = self._draw(env, 2)
                self.hole_cards[env, player, 0] = cards[0]
                self.hole_cards[env, player, 1] = cards[1]

        self.current_player[env] = self._next_eligible(env, self.dealer[env])

        if self.config.ante > 0:
            for player in self._eligible_players(env):
                self._bet(env, player, self.config.ante)
        self._post_blinds(env)
        self._reset_pending(env, exclude=self.last_aggressor[env])

    def _next_eligible(self, env: int, start: int) -> int:
        idx = start
        for _ in range(self.config.num_players):
            idx = (idx + 1) % self.config.num_players
            if not self.busted[env, idx]:
                return idx
        return start

    def _next_player(self, env: int, start: int) -> int:
        idx = start
        for _ in range(self.config.num_players):
            idx = (idx + 1) % self.config.num_players
            if not self.folded[env, idx] and not self.all_in[env, idx] and not self.busted[env, idx]:
                return idx
        return start

    def _eligible_players(self, env: int) -> List[int]:
        return [i for i in range(self.config.num_players) if not self.folded[env, i] and not self.busted[env, i]]

    def _actionable_players(self, env: int) -> List[int]:
        return [
            i
            for i in range(self.config.num_players)
            if not self.folded[env, i] and not self.all_in[env, i] and not self.busted[env, i]
        ]

    def _reset_pending(self, env: int, exclude: int = -1):
        self.pending[env, :] = False
        for player in self._actionable_players(env):
            if player != exclude:
                self.pending[env, player] = True

    def _post_blinds(self, env: int):
        if self.config.num_players == 2:
            sb_player = self.dealer[env]
            bb_player = self._next_player(env, sb_player)
        else:
            sb_player = self._next_player(env, self.dealer[env])
            bb_player = self._next_player(env, sb_player)
        self._bet(env, sb_player, self.config.small_blind)
        self._bet(env, bb_player, self.config.big_blind)
        self.street_bet[env, sb_player] = self.config.small_blind
        self.street_bet[env, bb_player] = self.config.big_blind
        self.current_bet[env] = self.config.big_blind
        self.min_raise[env] = self.config.big_blind
        self.last_aggressor[env] = bb_player
        self.current_player[env] = self._next_player(env, bb_player)

    def _bet(self, env: int, player: int, amount: int) -> int:
        amount = min(amount, int(self.stacks[env, player]))
        self.stacks[env, player] -= amount
        self.in_pot[env, player] += amount
        if self.stacks[env, player] == 0 and not self.folded[env, player]:
            self.all_in[env, player] = True
        return amount

    def _advance_round(self, env: int):
        self.raise_count[env] = 0
        self.current_bet[env] = 0
        self.min_raise[env] = self.config.big_blind
        self.last_aggressor[env] = -1
        self.street_bet[env, :] = 0
        self._reset_pending(env, exclude=-1)

        if self.round_index[env] == 0:
            cards = self._draw(env, 3)
            self.board[env, 0:3] = cards
        elif self.round_index[env] == 1:
            self.board[env, 3] = self._draw(env, 1)[0]
        elif self.round_index[env] == 2:
            self.board[env, 4] = self._draw(env, 1)[0]
        else:
            return

        self.round_index[env] += 1
        if self._actionable_players(env):
            self.current_player[env] = self._next_player(env, self.dealer[env])
        else:
            self.current_player[env] = self._next_eligible(env, self.dealer[env])

    def _apply_rake(self, env: int, pot: int, street_index: int | None) -> int:
        if self.config.rake_pct <= 0:
            return pot
        rake = int(pot * self.config.rake_pct)
        if self.config.rake_cap > 0:
            rake = min(rake, self.config.rake_cap)
        if self.config.rake_cap_per_hand > 0:
            rake = min(rake, max(0, self.config.rake_cap_per_hand - self.rake_taken_hand[env]))
        if street_index is not None and self.config.rake_cap_per_street > 0:
            rake = min(
                rake,
                max(
                    0,
                    self.config.rake_cap_per_street - self.rake_taken_street[env, street_index],
                ),
            )
        rake = max(0, rake)
        self.rake_taken_hand[env] += rake
        if street_index is not None:
            self.rake_taken_street[env, street_index] += rake
        return max(0, pot - rake)

    def _resolve_fold(self, env: int):
        active = self._eligible_players(env)
        if len(active) == 1:
            winner = active[0]
            pot = self._apply_rake(env, int(self.in_pot[env].sum()), street_index=min(self.round_index[env], 3))
            self.stacks[env, winner] += pot

    def _resolve_showdown(self, env: int):
        active = self._eligible_players(env)
        if not active:
            return

        contributions = self.in_pot[env].tolist()
        levels = sorted({c for c in contributions if c > 0})
        prev = 0
        for level in levels:
            pot = (level - prev) * sum(1 for c in contributions if c >= level)
            pot = self._apply_rake(env, pot, street_index=3)
            eligible = [i for i in active if contributions[i] >= level]
            if not eligible:
                prev = level
                continue
            scores = []
            board = [int(c) for c in self.board[env] if c >= 0]
            for player in eligible:
                hole = [int(c) for c in self.hole_cards[env, player] if c >= 0]
                scores.append((self.evaluator.evaluate(board, hole), player))
            best = min(scores, key=lambda x: x[0])[0]
            winners = [i for score, i in scores if score == best]
            share = pot // len(winners)
            remainder = pot - share * len(winners)
            for player in winners:
                self.stacks[env, player] += share
            if remainder > 0:
                winners_sorted = sorted(
                    winners, key=lambda i: (i - self.dealer[env]) % self.config.num_players
                )
                self.stacks[env, winners_sorted[0]] += remainder
            prev = level

    def _finish_hand(self, env: int):
        self.hands_left[env] -= 1
        for player in range(self.config.num_players):
            if self.stacks[env, player] <= 0:
                self.busted[env, player] = True
        eligible = [i for i in range(self.config.num_players) if not self.busted[env, i]]
        if self.hands_left[env] <= 0 or len(eligible) <= 1:
            self.episode_over[env] = True
            return
        self._start_hand(env, new_hand=True)

    def _record_action(self, env: int, player: int, action_type: int, bet_frac: float):
        self.hist_player[env] = np.roll(self.hist_player[env], -1)
        self.hist_action[env] = np.roll(self.hist_action[env], -1)
        self.hist_bet_frac[env] = np.roll(self.hist_bet_frac[env], -1)
        self.hist_round[env] = np.roll(self.hist_round[env], -1)
        self.hist_player[env, -1] = player
        self.hist_action[env, -1] = action_type
        self.hist_bet_frac[env, -1] = bet_frac
        self.hist_round[env, -1] = int(self.round_index[env])

    def legal_action_mask(self, env: int, player: int) -> np.ndarray:
        mask = np.zeros(3, dtype=np.float32)
        if self.folded[env, player] or self.all_in[env, player] or self.busted[env, player]:
            return mask
        to_call = max(0, int(self.current_bet[env] - self.street_bet[env, player]))
        if to_call > 0:
            mask[0] = 1.0
        mask[1] = 1.0
        can_raise = self.stacks[env, player] > to_call
        if self.config.max_raises_per_round > 0 and self.raise_count[env] >= self.config.max_raises_per_round:
            can_raise = False
        if can_raise:
            mask[2] = 1.0
        return mask

    def get_obs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        b = self.config.batch_size
        obs = np.zeros((b, self.obs_dim), dtype=np.float32)
        masks = np.zeros((b, 3), dtype=np.float32)
        for env in range(b):
            player = int(self.current_player[env])
            if self.episode_over[env]:
                masks[env] = 0.0
            else:
                masks[env] = self.legal_action_mask(env, player)
            for card in self.hole_cards[env, player]:
                if card >= 0:
                    obs[env, self.card_index[int(card)]] = 1.0
            for card in self.board[env]:
                if card >= 0:
                    obs[env, self.card_index[int(card)]] = 1.0
            offset = 52
            obs[env, offset] = float(self.in_pot[env].sum()) / self.config.big_blind
            obs[env, offset + 1] = float(self.current_bet[env]) / self.config.big_blind
            to_call = max(0, int(self.current_bet[env] - self.street_bet[env, player]))
            obs[env, offset + 2] = float(to_call) / self.config.big_blind
            obs[env, offset + 3] = float(self.round_index[env])
            obs[env, offset + 4] = float((player - self.dealer[env]) % self.config.num_players) / self.config.num_players
            offset += 5
            obs[env, offset : offset + self.config.num_players] = (
                self.stacks[env] / self.config.big_blind
            )
            offset += self.config.num_players
            obs[env, offset : offset + self.config.num_players] = (
                self.in_pot[env] / self.config.big_blind
            )
            offset += self.config.num_players
            obs[env, offset : offset + self.config.num_players] = (
                self.street_bet[env] / self.config.big_blind
            )
            offset += self.config.num_players
            obs[env, offset : offset + self.config.num_players] = self.folded[env].astype(np.float32)
            offset += self.config.num_players
            obs[env, offset : offset + self.config.num_players] = self.all_in[env].astype(np.float32)
            offset += self.config.num_players
            obs[env, offset : offset + self.config.num_players] = (
                (self.last_action[env] + 1) / 3.0
            )
            offset += self.config.num_players
            obs[env, offset : offset + self.config.num_players] = self.last_bet_frac[env]
            offset += self.config.num_players
            for idx in range(self.config.history_len):
                base = offset + idx * 4
                obs[env, base] = float(self.hist_action[env, idx] + 1) / 3.0
                obs[env, base + 1] = float(self.hist_bet_frac[env, idx])
                obs[env, base + 2] = (
                    float((self.hist_player[env, idx] - self.dealer[env]) % self.config.num_players)
                    / self.config.num_players
                    if self.hist_player[env, idx] >= 0
                    else 0.0
                )
                obs[env, base + 3] = float(self.hist_round[env, idx]) / 3.0 if self.hist_round[env, idx] >= 0 else 0.0

        return obs, masks, self.current_player.copy()

    def step(self, action_types: np.ndarray, bet_fracs: np.ndarray):
        b = self.config.batch_size
        for env in range(b):
            if self.episode_over[env]:
                continue

            if not self._actionable_players(env):
                while self.round_index[env] < 3:
                    self._advance_round(env)
                self.round_index[env] = 4
                self._resolve_showdown(env)
                self._finish_hand(env)
                continue

            player = int(self.current_player[env])
            if self.folded[env, player] or self.all_in[env, player] or self.busted[env, player]:
                self.current_player[env] = self._next_player(env, player)
                continue

            action_type = int(action_types[env])
            bet_frac = float(np.clip(bet_fracs[env], 0.0, 1.0))
            legal = self.legal_action_mask(env, player)
            to_call = max(0, int(self.current_bet[env] - self.street_bet[env, player]))
            actual_action = 1
            actual_bet_frac = 0.0

            if action_type == 0 and legal[0] > 0:
                self.folded[env, player] = True
                self.pending[env, player] = False
                actual_action = 0
            elif action_type == 2 and legal[2] > 0:
                max_raise = int(self.stacks[env, player] + to_call)
                min_raise = int(self.min_raise[env])
                raise_amount = min_raise
                if max_raise > min_raise:
                    raise_amount = int(min_raise + bet_frac * (max_raise - min_raise))
                total = to_call + raise_amount
                paid = self._bet(env, player, total)
                self.street_bet[env, player] += paid
                actual_action = 2
                actual_bet_frac = float(raise_amount) / max(1.0, float(max_raise))
                if paid > to_call:
                    actual_raise = paid - to_call
                    if self.street_bet[env, player] > self.current_bet[env]:
                        self.current_bet[env] = self.street_bet[env, player]
                    if actual_raise >= min_raise:
                        self.min_raise[env] = max(min_raise, actual_raise)
                        self.raise_count[env] += 1
                        self.last_aggressor[env] = player
                        self._reset_pending(env, exclude=player)
                    else:
                        self.pending[env, player] = False
                else:
                    self.pending[env, player] = False
            else:
                paid = self._bet(env, player, to_call)
                self.street_bet[env, player] += paid
                self.pending[env, player] = False

            self.last_action[env, player] = actual_action
            self.last_bet_frac[env, player] = actual_bet_frac
            self._record_action(env, player, actual_action, actual_bet_frac)

            if len(self._eligible_players(env)) <= 1:
                self._resolve_fold(env)
                self.round_index[env] = 4
                self._finish_hand(env)
                continue

            if not self._actionable_players(env):
                while self.round_index[env] < 3:
                    self._advance_round(env)
                self.round_index[env] = 4
                self._resolve_showdown(env)
                self._finish_hand(env)
                continue

            if not self.pending[env].any():
                if self.round_index[env] < 3:
                    self._advance_round(env)
                else:
                    self.round_index[env] = 4
                    self._resolve_showdown(env)
                    self._finish_hand(env)
                continue

            self.current_player[env] = self._next_player(env, player)

        return self.get_obs()

    def get_payoffs(self) -> np.ndarray:
        return self.stacks - self.starting_stacks
