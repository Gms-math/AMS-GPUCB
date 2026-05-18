# AMS-GPUCB

Optimized AlphaMapleSAT framework using Gaussian Process Upper Confidence Bound (GP-UCB) guided Monte Carlo Tree Search (MCTS) for Ramsey SAT cubing experiments.

---

## Overview

This repository contains an experimental extension of AlphaMapleSAT (AMS) integrating Bayesian Optimization methods into Monte Carlo Tree Search for combinatorial SAT solving and Ramsey theory experiments.

The primary goal is to investigate whether Gaussian Process Upper Confidence Bound (GP-UCB) guidance can reduce cubing runtime and improve branching efficiency on difficult Ramsey SAT instances such as:

- R(5,5)
- R(8,3)
- R(9,3)

The project compares:

1. Classical AMS/MCTS using UCT selection
2. AMS-GP-UCB hybrid search

---

## Mathematical Motivation

Classical MCTS uses the UCT (Upper Confidence Bound applied to Trees) formula:

U = Q(s,a) + c * sqrt(N(s)) / (1 + N(s,a))

where:

- Q(s,a) = average reward for branching action a
- N(s) = state visit count
- N(s,a) = edge/action visit count

This treats each branching action mostly independently.

AMS-GP-UCB augments this with Bayesian Optimization:

U = alpha * UCT + (1 - alpha) * (mu(x) + sqrt(beta) * sigma(x))

where:

- mu(x) = GP predicted branching quality
- sigma(x) = uncertainty estimate
- beta = exploration parameter

This allows the search to generalize structural information across similar SAT/Ramsey states rather than relearning each branch independently.

---

## Repository Structure

```text
main.py        Main experiment driver
MCTS.py        MCTS search logic (baseline or GP-UCB version)
gp_ucb.py      Gaussian Process helper implementation
Coach.py       Cubing/search execution loop
ksgraph/       Ramsey SAT game logic
requirements.txt Python package requirements
```

---

## Environment Setup

Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
```

Activate environment (Windows):

```bash
.venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
pip install scikit-learn psutil numpy wandb coloredlogs tqdm python-sat
```

---

## Small Benchmark Test

Run the included benchmark instance:

```bash
python -u main.py "constraints_17_c_100000_2_2_0_final.simp" -d 1 -m 136 -numMCTSSims 50 -o "gpucb_50.cubes" -prod
```

Expected output format:

```text
Time taken for cubing: ...
Reward: ...
Tool runtime: ...
```

---

## Preliminary Experimental Results

| Method | MCTS Sims | Cubing Time | Total Runtime | Reward |
|---|---:|---:|---:|---:|
| Baseline AMS/MCTS | 10 | 1.537 s | 3.024 s | 0.0328719723 |
| AMS-GP-UCB | 10 | 5.362 s | 8.864 s | 0.0328719723 |
| Baseline AMS/MCTS | 50 | 21.166 s | 22.935 s | 0.0328719723 |
| AMS-GP-UCB | 50 | 14.192 s | 15.632 s | 0.0328719723 |

At 50 MCTS simulations, AMS-GP-UCB reduced cubing runtime by approximately 33% while maintaining equivalent reward quality.

---

## HPC / Large Ramsey Experiments

The primary research goal of AMS-GPUCB is to scale Bayesian-guided cubing experiments toward larger Ramsey SAT instances using high-performance computing resources.

Current planned experiments include:

- R(5,5)
- R(8,3)
- R(9,3)

using larger MCTS simulation budgets and deeper cubing depths.

---

## Running AMS-GPUCB on HPC

The GP-UCB version uses:

- gp_ucb.py
- GP-UCB enabled MCTS.py

Before running large experiments, confirm the GP-UCB version of `MCTS.py` is active.

---

## Example GP-UCB Runs

### Medium simulation experiment

```bash
python -u main.py "constraints_17_c_100000_2_2_0_final.simp" -d 1 -m 136 -numMCTSSims 100 -o "gpucb_100.cubes" -prod
```

### Larger scaling experiment

```bash
python -u main.py "constraints_17_c_100000_2_2_0_final.simp" -d 2 -m 136 -numMCTSSims 200 -o "gpucb_200.cubes" -prod
```

### Deep cubing experiment

```bash
python -u main.py "constraints_17_c_100000_2_2_0_final.simp" -d 3 -m 136 -numMCTSSims 500 -o "gpucb_500.cubes" -prod
```

---

## Recommended HPC Metrics

For each experiment record:

- Cubing time
- Total runtime
- Reward
- Number of generated cubes
- SAT solver completion time
- Memory usage
- Scaling behavior as simulations increase

---

## Research Objective

The primary hypothesis is:

> Gaussian Process guided MCTS can reduce redundant combinatorial exploration in Ramsey SAT cubing by learning structural similarities between branching states.

The long-term objective is to determine whether Bayesian-guided cubing can improve large-scale Ramsey SAT computations relative to baseline AMS/MCTS.

---

## Repository

GitHub Repository:

https://github.com/Gms-math/AMS-GPUCB
