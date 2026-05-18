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
