"""
train.py
Double DQN agent for Chef's Hat Gym.

Variant 5 - Robustness and Generalisation:
Evaluates how well the agent generalises across opponent strategies,
random seeds, and hyperparameter configurations.

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
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gym
from ChefsHatGym.env import ChefsHatEnv

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Hyperparameters ────────────────────────────────────────────────────────────
TOTAL_GAMES   = 1000
GAMMA         = 0.95
LR            = 1e-3
HIDDEN_UNITS  = 256
STATE_DIM     = 228
ACTION_COUNT  = 200
BATCH_SIZE    = 64
BUFFER_CAP    = 50_000
TARGET_UPDATE = 200
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY     = 0.995


# ══════════════════════════════════════════════════════════════════════════════
# Q-Network
# ══════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """
    Two-hidden-layer Q-value network.
    Input  : normalised observation vector (STATE_DIM)
    Output : Q-value for each of the ACTION_COUNT actions
    """
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
# Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Fixed-size circular experience replay buffer with masked next-state info."""

    def __init__(self, capacity=BUFFER_CAP):
        from collections import deque
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done, next_valid_mask):
        self.buf.append((s, a, r, s2, done, next_valid_mask))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d, nvm = zip(*batch)
        return (
            np.array(s,   dtype=np.float32),
            np.array(a,   dtype=np.int64),
            np.array(r,   dtype=np.float32),
            np.array(s2,  dtype=np.float32),
            np.array(d,   dtype=np.float32),
            np.array(nvm, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ══════════════════════════════════════════════════════════════════════════════
# Double DQN Agent
# ══════════════════════════════════════════════════════════════════════════════

class DQNAgent:
    """
    Double DQN agent designed for robustness and generalisation.

    State representation:
        Raw observation vector (~228 values) from ChefsHatGym,
        normalised to [0, 1] by dividing by the maximum value.
        No hand-crafted features are used so that generalisation
        claims are not confounded by domain-specific engineering.

    Action handling:
        The environment provides a binary action mask each turn.
        Invalid actions are set to a large negative value before argmax
        so the agent never selects an illegal move during training or evaluation.
        During exploration, sampling is restricted to valid actions only.

    Reward:
        Raw terminal reward mapped from finishing rank to [-1, +1].
        No intermediate shaping is applied. Robustness to sparse
        rewards is addressed through Double DQN and a large replay
        buffer rather than manual reward engineering.

    Algorithm:
        Double DQN decouples action selection (policy net) from
        action evaluation (target net), reducing Q-value overestimation
        which is harmful when rewards are sparse and delayed.
        For masked-action environments, next-state action masking is
        applied during target computation.
    """

    def __init__(self, seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.opt     = optim.Adam(self.policy.parameters(), lr=LR)
        self.loss_fn = nn.SmoothL1Loss()
        self.buf     = ReplayBuffer()

        self.epsilon = EPS_START
        self.steps   = 0
        self.updates = 0
        self._last_s = None
        self._last_a = None

    def act(self, obs, valid_mask):
        """Select action using epsilon-greedy with action masking."""
        s = self._preprocess(obs)
        self._last_s = s

        valid_mask = np.asarray(valid_mask, dtype=np.float32)
        valid = np.where(valid_mask == 1)[0]

        if len(valid) == 0:
            valid = np.arange(ACTION_COUNT)

        if random.random() < self.epsilon:
            a = int(np.random.choice(valid))
        else:
            with torch.no_grad():
                q = self.policy(
                    torch.as_tensor(s, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
                ).squeeze(0).cpu().numpy()
            mask_penalty = np.where(valid_mask == 1, 0.0, -1e9)
            a = int(np.argmax(q + mask_penalty))

        self._last_a = a
        return a

    def observe(self, next_obs, reward, done, next_valid_mask):
        """Store transition."""
        s2 = self._preprocess(next_obs)
        if self._last_s is not None and self._last_a is not None:
            self.buf.push(
                self._last_s, self._last_a, reward, s2,
                float(done),
                np.asarray(next_valid_mask, dtype=np.float32),
            )
        self.steps += 1
        if done:
            self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
            self._last_s = None
            self._last_a = None

    def _train_step(self):
        """One gradient update using a random mini-batch."""
        if len(self.buf) < BATCH_SIZE:
            return None

        s, a, r, s2, d, nvm = self.buf.sample(BATCH_SIZE)

        s   = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        a   = torch.as_tensor(a,  dtype=torch.long,    device=self.device).unsqueeze(1)
        r   = torch.as_tensor(r,  dtype=torch.float32, device=self.device)
        s2  = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d   = torch.as_tensor(d,  dtype=torch.float32, device=self.device)
        nvm = torch.as_tensor(nvm, dtype=torch.float32, device=self.device)

        next_mask_penalty = torch.where(
            nvm > 0.5,
            torch.zeros_like(nvm),
            torch.full_like(nvm, -1e9),
        )

        q = self.policy(s).gather(1, a).squeeze(1)

        with torch.no_grad():
            best_a  = (self.policy(s2) + next_mask_penalty).argmax(dim=1, keepdim=True)
            q_next  = (self.target(s2) + next_mask_penalty).gather(1, best_a).squeeze(1)
            q_target = r + GAMMA * q_next * (1 - d)

        loss = self.loss_fn(q, q_target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.opt.step()

        self.updates += 1
        if self.updates % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.policy.state_dict())

        return float(loss.item())

    def _preprocess(self, obs):
        obs = np.array(obs, dtype=np.float32)
        m = obs.max()
        return obs / m if m > 0 else obs


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def random_play(obs):
    """Baseline: uniformly random valid action."""
    v = np.where(np.asarray(obs[28:]) == 1)[0]
    act_arr = np.zeros(ACTION_COUNT, dtype=np.float32)
    act_arr[int(np.random.choice(v)) if len(v) > 0 else ACTION_COUNT - 1] = 1.0
    return act_arr


def terminal_reward(score):
    return {3: 1.0, 2: 0.2, 1: -0.4, 0: -1.0}.get(int(score), -1.0)


def extract_valid_mask(obs):
    """obs[28:28+ACTION_COUNT] is the binary valid-action mask."""
    mask = np.asarray(obs[28:28 + ACTION_COUNT], dtype=np.float32)
    if mask.shape[0] != ACTION_COUNT:
        raise ValueError(
            f"Expected valid mask length {ACTION_COUNT}, got {mask.shape[0]}."
        )
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def run_training(n_games=TOTAL_GAMES, seed=SEED, save_dir="outputs/"):
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(os.path.dirname(save_dir.rstrip("/\\")), "logs")
    os.makedirs(log_dir, exist_ok=True)

    dqn  = DQNAgent(seed=seed)
    seat = 0
    wins, scores, q_losses, epsilons = [], [], [], []
    win_count = 0

    env = gym.make("chefshat-v1")

    print("=" * 55)
    print(" Double DQN | Variant 5: Robustness & Generalisation")
    print(f" Author  : Sai Sumanth Desagani | ID: 16464285")
    print(f" Matches : {n_games} | Seed: {seed}")
    print("=" * 55)

    for g in range(n_games):
        env.startExperiment(
            gameType=ChefsHatEnv.GAMETYPE["MATCHES"],
            stopCriteria=1, maxRounds=-1,
            playerNames=["DQN_Agent", "Rand_A", "Rand_B", "Rand_C"],
            logDirectory=log_dir, verbose=0,
            saveDataset=True, saveLog=True,
        )
        try:
            obs, _ = env.reset(seed=seed + g)
        except TypeError:
            obs, _ = env.reset()

        done = False
        ep_loss, ep_updates = 0.0, 0

        while not done:
            if env.currentPlayer == seat:
                valid_mask = extract_valid_mask(obs)
                act_idx    = dqn.act(obs, valid_mask)
                act_arr    = np.zeros(ACTION_COUNT, dtype=np.float32)
                act_arr[act_idx] = 1.0

                next_obs, _, done, _, info = env.step(act_arr)
                r = terminal_reward(info["Match_Score"][seat]) if done else 0.0
                next_valid_mask = (np.ones(ACTION_COUNT, dtype=np.float32)
                                   if done else extract_valid_mask(next_obs))

                dqn.observe(next_obs, r, done, next_valid_mask)
                l = dqn._train_step()
                if l is not None:
                    ep_loss += l
                    ep_updates += 1
                obs = next_obs
            else:
                obs, _, done, _, info = env.step(random_play(obs))

        sc = info["Match_Score"][seat]
        won = 1 if sc == 3 else 0
        win_count += won
        wins.append(won)
        scores.append(int(sc))
        q_losses.append(ep_loss / max(ep_updates, 1))
        epsilons.append(float(dqn.epsilon))

        if (g + 1) % 50 == 0:
            wr = np.mean(wins[-50:]) * 100
            print(f"  Match {g+1:>4}/{n_games} | WR(50): {wr:5.1f}% | "
                  f"ε={dqn.epsilon:.3f} | Loss: {q_losses[-1]:.4f}")

    # Save model
    model_path = os.path.join(save_dir, "dqn_agent.pth")
    torch.save({"policy": dqn.policy.state_dict(),
                "target": dqn.target.state_dict(),
                "epsilon": dqn.epsilon,
                "steps": dqn.steps,
                "updates": dqn.updates}, model_path)

    # Save metrics
    data = {
        "wins": [int(x) for x in wins],
        "scores": [int(x) for x in scores],
        "q_losses": [float(x) for x in q_losses],
        "epsilon": [float(x) for x in epsilons],
        "total_wins": int(win_count),
        "win_rate": round(float(win_count / n_games), 4),
        "config": {
            "author": "Sai Sumanth Desagani", "id": "16464285",
            "variant": 5, "n_games": int(n_games), "seed": int(seed),
            "gamma": GAMMA, "lr": LR, "eps_decay": EPS_DECAY,
            "batch_size": BATCH_SIZE, "buffer_cap": BUFFER_CAP,
            "target_update": TARGET_UPDATE,
        },
    }
    metrics_path = os.path.join(save_dir, "dqn_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(data, f, indent=2)

    _plot_training(wins, scores, q_losses, epsilons, save_dir)

    print(f"\n  Training complete. Win rate: {win_count / n_games * 100:.1f}%")
    print(f"  Model   → {model_path}")
    print(f"  Metrics → {metrics_path}")
    return dqn, data


# ══════════════════════════════════════════════════════════════════════════════
# 6-panel training plot
# ══════════════════════════════════════════════════════════════════════════════

def _plot_training(wins, scores, q_losses, epsilons, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    N, W = len(wins), 20

    rw = [np.mean(wins[max(0, i-W+1):i+1]) * 100 for i in range(N)]
    axes[0,0].plot(rw, color="steelblue", linewidth=1.3)
    axes[0,0].set_title("Rolling Win Rate (window=20)"); axes[0,0].set_ylabel("%")
    axes[0,0].set_xlabel("Match"); axes[0,0].grid(True, alpha=0.3)

    rs = [np.mean(scores[max(0, i-W+1):i+1]) for i in range(N)]
    axes[0,1].plot(scores, alpha=0.2, color="gold")
    axes[0,1].plot(rs, color="sienna", linewidth=1.5, label="Smoothed")
    axes[0,1].set_title("Match Scores"); axes[0,1].set_ylabel("Score")
    axes[0,1].set_xlabel("Match"); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    axes[0,2].plot(q_losses, color="indigo", alpha=0.5)
    if len(q_losses) > 30:
        sm = np.convolve(q_losses, np.ones(30)/30, mode="valid")
        axes[0,2].plot(range(29, N), sm, color="indigo", linewidth=1.5)
    axes[0,2].set_title("TD Loss (Huber)"); axes[0,2].set_ylabel("Loss")
    axes[0,2].set_xlabel("Match"); axes[0,2].grid(True, alpha=0.3)

    axes[1,0].plot(epsilons, color="tomato", linewidth=1.5)
    axes[1,0].axhline(EPS_END, color="grey", linestyle="--", alpha=0.5, label="ε_min")
    axes[1,0].set_title("Epsilon Decay"); axes[1,0].set_ylabel("ε")
    axes[1,0].set_xlabel("Match"); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    cum = [sum(wins[:i+1]) / (i+1) * 100 for i in range(N)]
    axes[1,1].plot(cum, color="darkcyan", linewidth=1.5)
    axes[1,1].axhline(25, color="gray", linestyle="--", alpha=0.7, label="Random (~25%)")
    axes[1,1].set_title("Cumulative Win Rate"); axes[1,1].set_ylabel("%")
    axes[1,1].set_xlabel("Match"); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    unique, counts = np.unique(scores, return_counts=True)
    axes[1,2].bar([str(u) for u in unique], counts, color="slategray")
    axes[1,2].set_title("Score Distribution")
    axes[1,2].set_xlabel("Score (0=last, 3=first)"); axes[1,2].set_ylabel("Count")
    axes[1,2].grid(axis="y", alpha=0.3)

    plt.suptitle("DQN Training — Variant 5: Robustness & Generalisation\n"
                 "Sai Sumanth Desagani | 16464285", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "dqn_training_plots.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Training plots → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train Double DQN on Chef's Hat (Variant 5)")
    p.add_argument("--matches",  type=int, default=TOTAL_GAMES)
    p.add_argument("--seed",     type=int, default=SEED)
    p.add_argument("--save_dir", default="outputs/")
    a = p.parse_args()
    run_training(n_games=a.matches, seed=a.seed, save_dir=a.save_dir)
