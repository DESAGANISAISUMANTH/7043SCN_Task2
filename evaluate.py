"""
evaluate.py
Tests the trained Double DQN agent against a random baseline.
Reports win rate and average score over N matches each.

Author: Sai Sumanth Desagani
Student ID: 16464285
Variant: Robustness and Generalisation (ID mod 7 = 5)
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gym
from ChefsHatGym.env import ChefsHatEnv

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

STATE_DIM    = 228
ACTION_COUNT = 200
HIDDEN_UNITS = 256
N_EVAL       = 200


# ══════════════════════════════════════════════════════════════════════════════
# Network (must match train.py)
# ══════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    def __init__(self, n_in=STATE_DIM, n_out=ACTION_COUNT, n_h=HIDDEN_UNITS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_h), nn.ReLU(),
            nn.Linear(n_h, n_h),  nn.ReLU(),
            nn.Linear(n_h, n_out),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers (consistent with train.py)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(obs):
    """Normalise observation to [0, 1] exactly like train.py."""
    obs = np.asarray(obs, dtype=np.float32)
    m = obs.max()
    return obs / m if m > 0 else obs


def extract_valid_mask(obs):
    """
    Chef's Hat observation format:
    obs[28:28+ACTION_COUNT] is the binary valid-action mask.
    """
    mask = np.asarray(obs[28:28 + ACTION_COUNT], dtype=np.float32)
    if mask.shape[0] != ACTION_COUNT:
        raise ValueError(
            f"Expected valid mask length {ACTION_COUNT}, got {mask.shape[0]}. "
            "Check observation format and constants."
        )
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# Action helpers
# ══════════════════════════════════════════════════════════════════════════════

def random_play(obs):
    """Uniformly random valid action — used as baseline."""
    valid_mask = extract_valid_mask(obs)
    v = np.where(valid_mask == 1)[0]
    a = np.zeros(ACTION_COUNT, dtype=np.float32)
    a[int(np.random.choice(v)) if len(v) > 0 else (ACTION_COUNT - 1)] = 1.0
    return a


def model_play(net, obs, device="cpu"):
    """Greedy action from trained network with action masking."""
    valid_mask = extract_valid_mask(obs)
    v = np.where(valid_mask == 1)[0]
    a = np.zeros(ACTION_COUNT, dtype=np.float32)
    if len(v) == 0:
        a[ACTION_COUNT - 1] = 1.0
        return a

    x = preprocess(obs)
    with torch.no_grad():
        q = net(
            torch.as_tensor(x, dtype=torch.float32,
                             device=device).unsqueeze(0)
        ).squeeze(0).cpu().numpy()

    mask_penalty = np.where(valid_mask == 1, 0.0, -1e9)
    a[int(np.argmax(q + mask_penalty))] = 1.0
    return a


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation runner
# ══════════════════════════════════════════════════════════════════════════════

def run_eval(action_fn, n_matches, tag, log_dir, seed=SEED):
    """
    Run n_matches using action_fn.
    Returns (scores, win_rate%, avg_score).
    action_fn signature: action_fn(obs) -> one-hot action array
    """
    ld = os.path.join(log_dir, f"ev_{tag}")
    os.makedirs(ld, exist_ok=True)

    env = gym.make("chefshat-v1")
    scores, wins = [], 0

    for i in range(n_matches):
        env.startExperiment(
            gameType=ChefsHatEnv.GAMETYPE["MATCHES"],
            stopCriteria=1, maxRounds=-1,
            playerNames=["Eval_P", "B1", "B2", "B3"],
            logDirectory=ld, verbose=0,
            saveDataset=True, saveLog=True,
        )
        try:
            obs, _ = env.reset(seed=seed + i)
        except TypeError:
            obs, _ = env.reset()

        done = False
        while not done:
            act = action_fn(obs) if env.currentPlayer == 0 else random_play(obs)
            obs, _, done, _, info = env.step(act)

        s = int(info["Match_Score"][0])
        scores.append(s)
        if s == 3:
            wins += 1

        if (i + 1) % 50 == 0:
            print(f"    [{tag}] {i+1}/{n_matches} done")

    win_rate  = round(wins / n_matches * 100.0, 1)
    avg_score = round(float(np.mean(scores)), 2)
    return scores, win_rate, avg_score


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    base    = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "outputs")
    log_dir = os.path.join(base, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ── Load trained model ────────────────────────────────────────────────────
    model_path = os.path.join(out_dir, "dqn_agent.pth")
    if not os.path.exists(model_path):
        print("Model not found. Run train.py first.")
        return

    checkpoint = torch.load(model_path, map_location="cpu")
    net = QNetwork()
    net.load_state_dict(checkpoint["policy"])
    net.eval()

    print("=" * 55)
    print(" Evaluation — Sai Sumanth Desagani | 16464285")
    print(f" {N_EVAL} matches per condition")
    print("=" * 55)

    # ── Run evaluations ───────────────────────────────────────────────────────
    print("\n[1/2] DQN Agent vs Random opponents...")
    sc_dqn, wr_dqn, avg_dqn = run_eval(
        action_fn=lambda o: model_play(net, o, device="cpu"),
        n_matches=N_EVAL, tag="dqn_agent",
        log_dir=log_dir, seed=SEED,
    )

    print("\n[2/2] Random Baseline vs Random opponents...")
    sc_r, wr_r, avg_r = run_eval(
        action_fn=random_play,
        n_matches=N_EVAL, tag="random_baseline",
        log_dir=log_dir, seed=SEED + 10_000,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Results")
    print("=" * 55)
    print(f"  DQN Agent (Variant 5) : WR={wr_dqn:.1f}%  Avg={avg_dqn:.2f}")
    print(f"  Random Baseline       : WR={wr_r:.1f}%   Avg={avg_r:.2f}")
    print("=" * 55)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results = {
        "author":  "Sai Sumanth Desagani",
        "id":      "16464285",
        "variant": 5,
        "n_eval":  N_EVAL,
        "seed":    SEED,
        "dqn_agent": {
            "win_rate":  float(wr_dqn),
            "avg_score": float(avg_dqn),
            "scores":    [int(x) for x in sc_dqn],
        },
        "random_baseline": {
            "win_rate":  float(wr_r),
            "avg_score": float(avg_r),
            "scores":    [int(x) for x in sc_r],
        },
    }
    results_path = os.path.join(out_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {results_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    names  = ["DQN Agent\n(Variant 5)", "Random\nBaseline"]
    colors = ["steelblue", "gray"]

    a1.bar(names, [wr_dqn, wr_r], color=colors)
    a1.set_ylabel("Win Rate (%)")
    a1.set_title("Win Rate Comparison")
    a1.grid(axis="y", alpha=0.3)
    for i, v in enumerate([wr_dqn, wr_r]):
        a1.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

    a2.bar(names, [avg_dqn, avg_r], color=colors)
    a2.set_ylabel("Avg Score")
    a2.set_title("Average Score")
    a2.grid(axis="y", alpha=0.3)
    for i, v in enumerate([avg_dqn, avg_r]):
        a2.text(i, v + 0.01, f"{v:.2f}", ha="center", fontweight="bold")

    plt.suptitle("Evaluation — Variant 5: Robustness & Generalisation\n"
                 "Sai Sumanth Desagani | 16464285", fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "eval_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved      → {plot_path}")


if __name__ == "__main__":
    main()
