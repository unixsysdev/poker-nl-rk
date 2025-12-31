import argparse
from typing import Optional

import numpy as np
import torch
from torch.distributions import Categorical

import pyspiel

from poker_rl.alpha_policy import PolicyNet
from poker_rl.solvers import CFRPolicy


def select_action(policy, obs, legal_actions, device, greedy):
    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
    logits, _ = policy(obs_t)
    mask = torch.full_like(logits, -1e9)
    mask[:, legal_actions] = 0.0
    logits = logits + mask
    if greedy:
        return int(torch.argmax(logits, dim=-1).item())
    dist = Categorical(logits=logits)
    return int(dist.sample().item())


def play_episode(game, policy, opponent, device, rng, seat_policy, greedy):
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            state.apply_action(int(rng.choice(actions, p=probs)))
            continue

        player = state.current_player()
        legal_actions = state.legal_actions(player)
        obs = np.asarray(state.information_state_tensor(player), dtype=np.float32)

        if player == seat_policy:
            action = select_action(policy, obs, legal_actions, device, greedy)
        else:
            if isinstance(opponent, CFRPolicy):
                action = opponent.action(state)
            else:
                action = int(rng.choice(legal_actions))
        state.apply_action(action)

    return float(state.returns()[seat_policy])


def evaluate(
    game_name,
    model_path,
    opponent_type,
    episodes,
    greedy,
    device,
    cfr_iterations,
    cfr_algorithm,
    hidden_size,
):
    game = pyspiel.load_game(game_name)
    obs_dim = game.information_state_tensor_size()
    action_dim = game.num_distinct_actions()

    policy = PolicyNet(obs_dim, action_dim, hidden_size=hidden_size).to(device)
    state = torch.load(model_path, map_location=device)
    policy.load_state_dict(state)
    policy.eval()

    opponent: Optional[object] = None
    if opponent_type == "cfr":
        opponent = CFRPolicy(game_name, iterations=cfr_iterations, algorithm=cfr_algorithm)

    rng = np.random.default_rng(42)
    scores = []
    for i in range(episodes):
        seat_policy = i % 2
        scores.append(play_episode(game, policy, opponent, device, rng, seat_policy, greedy))
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Leduc policy.")
    parser.add_argument("--game", default="leduc_poker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--opponent", choices=["random", "cfr"], default="cfr")
    parser.add_argument("--cfr-iterations", type=int, default=1000)
    parser.add_argument("--cfr-algorithm", choices=["cfr", "cfr_plus"], default="cfr")
    parser.add_argument("--greedy", action="store_true", default=False)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    score = evaluate(
        args.game,
        args.model,
        args.opponent,
        args.episodes,
        args.greedy,
        args.device,
        args.cfr_iterations,
        args.cfr_algorithm,
        args.hidden_size,
    )
    print(f"avg_return={score:.4f}")


if __name__ == "__main__":
    main()
