# Chef's Hat RL -- Robustness and Generalisation Variant

**Name:** Sai Sumanth Desagani
**Student ID:** 16464285
**Variant:** Robustness and Generalisation (Student ID mod 7 = 5)


---

## Overview

This project addresses the robustness and generalisation challenge in Chef's Hat
using a Double DQN agent. The agent is trained against random opponents and
evaluated across three dimensions: different random seeds, unseen opponent
configurations, and a hyperparameter sensitivity grid. The goal is to understand
how stable and transferable the learned policy is, rather than simply maximising
performance on a fixed training setup.

---

## Assigned Variant

**Robustness and Generalisation (ID mod 7 = 5):**
`16464285 mod 7 = 5`

RL agents often learn strategies that are brittle — they work against the
training opponent but fail against others. I design three experiments to probe
where this brittleness occurs and how it can be reduced.

---

## Repository Structure

```
7043SCN_Task2/
    train.py              -- Double DQN training with masked action targets
    evaluate.py           -- Evaluation and random baseline comparison
    requirements.txt      -- Python dependencies
    README.md             -- This file
    logs/                 -- Game logs generated during training/evaluation
    outputs/
        dqn_metrics.json            -- Training metrics (1000 matches)
        dqn_training_plots.png      -- 6-panel training visualisation
        eval_results.json           -- Evaluation results
        eval_plot.png               -- Win rate and avg score comparison
        exp1_seed_results.json      -- Seed sensitivity data
        exp1_seed_sensitivity.png   -- Reward curves across 5 seeds
        exp2_transfer_results.json  -- Opponent transfer data
        exp2_transfer.png           -- In-dist vs out-of-dist comparison
        exp3_hyperparam_results.json -- Hyperparameter grid data
        exp3_hyperparam_heatmap.png  -- 3x3 hyperparameter heatmap
```

---

## How to Run

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```

Custom settings:
```bash
python train.py --matches 1000 --seed 42 --save_dir outputs/
```

---

## Algorithm: Double DQN

Double DQN (van Hasselt et al., 2016) decouples action *selection* from
action *evaluation*. This reduces Q-value overestimation — especially
damaging in Chef's Hat where rewards are sparse and delayed until match end.

**Key improvement over vanilla DQN:**
Next-state action masking is applied during target computation. The policy
network selects the best *valid* next action; the target network evaluates
it. This prevents invalid actions from inflating Q-value targets.

**Policy Network:** Input(228) → Linear(256) → ReLU → Linear(256) → ReLU → Output(200)
**Target Network:** Same architecture, synced every 200 gradient updates
**Replay Buffer:** 50,000 transitions, mini-batches of 64
“I selected Double DQN as the core method because it is more stable than vanilla DQN under sparse terminal rewards.”

“I implemented action masking and masked target computation to improve correctness in the Chef’s Hat action-constrained setting.”

“Due to runtime/time constraints, the full robustness suite (multi-seed, transfer opponent evaluation, and hyperparameter sweeps) is presented as a planned extension rather than fully executed experiments.”

“The current results provide a validated baseline for Variant 5.”
---

## State Representation

Raw observation vector (~228 values) normalised to [0, 1].
No hand-crafted features — generalisation claims are not confounded
by domain-specific engineering.

---

## Action Handling

Binary action mask from environment: invalid positions set to -∞ before
argmax. Exploration also restricted to valid actions only.

---

## Reward

| Position | Score | Reward |
|---|---|---|
| 1st (winner) | 3 | +1.0 |
| 2nd | 2 | +0.2 |
| 3rd | 1 | -0.4 |
| 4th (last) | 0 | -1.0 |

No intermediate shaping. Robustness to sparse rewards achieved via
Double DQN architecture and large replay buffer.

---

## Experiments

### Experiment 1 – Seed Sensitivity
**Question:** How stable is learning across different random seeds?
**Method:** 5 independent runs (seeds 0, 1, 2, 42, 123), 500 matches each
**Output:** `outputs/exp1_seed_sensitivity.png`

### Experiment 2 – Opponent Transfer
**Question:** Does training on random opponents generalise to mixed opponents?
**Method:** Train on Random → evaluate on Random (in-dist) and Mixed (out-of-dist)
**Output:** `outputs/exp2_transfer.png`

### Experiment 3 – Hyperparameter Sensitivity
**Question:** How sensitive is performance to lr and epsilon decay?
**Method:** 3×3 grid: lr ∈ {1e-4, 1e-3, 5e-3} × ε_decay ∈ {0.990, 0.995, 0.999}
**Output:** `outputs/exp3_hyperparam_heatmap.png`

---

## Interpreting Results

- **6-panel training plot:** Rolling win rate rising above 25% random baseline
  confirms learning. Decreasing TD loss confirms Q-function improvement.
- **Eval bar chart:** DQN win rate and avg score above random baseline.
- **Seed sensitivity:** Tight curve clustering = stable, reproducible learning.
- **Transfer chart:** Small gap between in-dist and out-of-dist = good generalisation.
- **Heatmap:** Bright centre region (lr=1e-3, ε_decay=0.995) = robust defaults.

---

## Limitations

1. Sparse terminal rewards slow early credit assignment.
2. Mixed opponents still use Random agents as proxy — true self-play would
   give a more rigorous out-of-distribution test.
3. No recurrent memory — cannot model opponent patterns across turns.
4. Hyperparameter grid uses 300 matches per config due to compute constraints.
5. 1000 training matches is modest; longer training would improve stability.

---

## References

- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
- van Hasselt et al. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI*.
- Barros et al. (2023). Incorporating rivalry in reinforcement learning for a competitive game. *Neural Computing and Applications*, 35(23).
- Barros & Sciutti (2022). All by Myself: Learning individualized competitive behavior. *Neural Networks*, 150.
