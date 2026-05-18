# AMS-GPUCB

Optimized AlphaMapleSAT framework using Gaussian Process Upper Confidence Bound (GP-UCB) guided Monte Carlo Tree Search (MCTS) for Ramsey SAT cubing experiments.

---

## Overview

AMS-GPUCB is an experimental extension of AlphaMapleSAT (AMS) integrating Bayesian Optimization methods into Monte Carlo Tree Search for combinatorial SAT solving and Ramsey theory experiments.

The primary objective is to investigate whether Gaussian Process Upper Confidence Bound (GP-UCB) guidance can reduce cubing runtime and improve branching efficiency on difficult Ramsey SAT instances such as:

- R(5,5)
- R(8,3)
- R(9,3)

The project compares:

1. Classical AMS/MCTS using UCT (Upper Confidence Bound applied to Trees)
2. AMS-GP-UCB hybrid search

---

## Mathematical Motivation

Classical Monte Carlo Tree Search uses the UCT formula:

U = Q(s,a) + c * sqrt(N(s)) / (1 + N(s,a))

where:

- Q(s,a) = average reward for branching action a
- N(s) = state visit count
- N(s,a) = edge/action visit count

This approach learns branching decisions primarily through local search statistics.

AMS-GP-UCB augments this with Bayesian Optimization:

U = alpha * UCT + (1 - alpha) * (mu(x) + sqrt(beta) * sigma(x))

where:

- mu(x) = GP predicted branching quality
- sigma(x) = uncertainty estimate
- beta = exploration parameter

The GP-UCB model attempts to generalize structural information across similar SAT/Ramsey states rather than relearning each branch independently.

---

## Repository Structure

```text
main.py        Main experiment driver
MCTS.py        MCTS search logic (GP-UCB enabled version)
gp_ucb.py      Gaussian Process helper implementation
Coach.py       Cubing/search execution loop
ksgraph/       Ramsey SAT game logic
requirements.txt Python package requirements
```

---

## Environment Setup

Create a Python virtual environment:

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

Planned Ramsey experiments include:

- R(5,5)
- R(8,3)
- R(9,3)

using:
- deeper cubing depths
- larger branching budgets
- increased MCTS simulation counts
- HPC parallelization

---

## Running Large Ramsey Experiments on HPC

The current repository contains:
- the AMS-GP-UCB framework
- benchmark experiments
- GP-UCB enabled MCTS implementation

Large Ramsey SAT encoding files are NOT currently included in the repository.

Examples of expected Ramsey SAT encoding files:

```text
R55.simp
R83.simp
R93.simp
```

These Ramsey SAT encodings must either:
- be generated separately using Ramsey SAT encoders, or
- be obtained from existing Ramsey SAT benchmark datasets

before launching large-scale HPC experiments.

Once obtained, the SAT instance files should be placed in the same directory as:

```text
main.py
```

before running the experiments.

---

## Example GP-UCB HPC Execution Commands

The following commands are intended for large-scale GP-UCB guided cubing experiments.

---

### R(5,5) Medium-Scale Experiment

Tests moderate cubing depth and simulation count for preliminary R(5,5) scaling behavior.

```bash
python -u main.py "R55.simp" -d 2 -m 500 -numMCTSSims 200 -o "R55_gpucb.cubes" -prod
```

---

### R(8,3) Large-Scale Experiment

Increases branching literals and simulation budget for larger Ramsey SAT exploration.

```bash
python -u main.py "R83.simp" -d 3 -m 1000 -numMCTSSims 500 -o "R83_gpucb.cubes" -prod
```

---

### R(9,3) HPC Intensive Experiment

Designed for large HPC runs with deeper cubing and heavy MCTS exploration.

```bash
python -u main.py "R93.simp" -d 3 -m 1500 -numMCTSSims 1000 -o "R93_gpucb.cubes" -prod
```

---

## Parameter Descriptions

- `-d`
  controls cubing depth

- `-m`
  controls the number of branching literals considered

- `-numMCTSSims`
  controls the Monte Carlo Tree Search simulation budget

Larger Ramsey instances generally require:
- deeper cubing
- larger simulation counts
- increased memory allocation
- HPC parallelization

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

## Current Status

The current repository demonstrates:
- a functional AMS-GP-UCB prototype
- preliminary benchmark comparisons
- scalable GP-UCB guided cubing experiments

Large Ramsey experiments such as:
- R(5,5)
- R(8,3)
- R(9,3)

require corresponding SAT encodings and future HPC experimentation.

---

## Repository

https://github.com/Gms-math/AMS-GPUCB
