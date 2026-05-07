import logging
import math
import random


import numpy as np
import psutil
import wandb


from gp_ucb import GPUCBHelper


EPS = 1e-8
random.seed(42)
np.random.seed(42)


log = logging.getLogger(__name__)




class MCTS:
    def __init__(self, nnet, args, all_logging_data, nn_iteration):
        self.nn_iteration = nn_iteration
        self.nnet = nnet
        self.args = args


        self.Qsa = {}
        self.Bsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}


        self.gp_helper = GPUCBHelper(
            beta=self.args["gp_beta"] if "gp_beta" in self.args else 2.0
        )
        self.GPsa = {}


        self.data = []
        self.all_logging_data = all_logging_data
        self.cache_data = {}


    def getActionProb(self, game, board, temp=0, verbose=False):
        canonicalBoard = game.getCanonicalForm(board)


        if verbose:
            print("RAM memory % used:", psutil.virtual_memory()[2])
            print("RAM Used (GB):", psutil.virtual_memory()[3] / 1e9)


        for sim in range(self.args.numMCTSSims):
            if self.args.debugging:
                log.info("MCTS Simulation #{}".format(sim))
            self.search(game, canonicalBoard, verbose=verbose)


        s = game.stringRepresentation(canonicalBoard)
        valids = game.getValidMoves(canonicalBoard)


        counts = [self.Nsa.get((s, a), 0) for a in range(game.getActionSize())]
        maxvals = [self.Bsa.get((s, a), -float("inf")) for a in range(game.getActionSize())]


        if temp == 0:
            valid_candidates = [a for a in range(game.getActionSize()) if valids[a]]


            # Prefer visited valid actions with Bsa values
            visited_valid_candidates = [
                a for a in valid_candidates if (s, a) in self.Bsa
            ]


            if visited_valid_candidates:
                bestA = max(visited_valid_candidates, key=lambda a: self.Bsa[(s, a)])
            else:
                # Fallback: choose valid action with largest policy score
                bestA = max(valid_candidates, key=lambda a: self.Ps[s][a])


            probs = [0] * game.getActionSize()
            probs[bestA] = 1
            return probs


        counts = [counts[a] if valids[a] else 0 for a in range(game.getActionSize())]
        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))


        if counts_sum == 0:
            valid_candidates = [a for a in range(game.getActionSize()) if valids[a]]
            probs = [0] * game.getActionSize()
            for a in valid_candidates:
                probs[a] = 1.0 / len(valid_candidates)
            return probs


        probs = [x / counts_sum for x in counts]
        return probs


    def search(self, game, canonicalBoard, verbose=False, level=0):
        s = game.stringRepresentation(canonicalBoard)


        if s not in self.Es:
            self.Es[s] = game.getGameEndedMCTS(canonicalBoard)


        if hasattr(canonicalBoard, "is_done") and canonicalBoard.is_done():
            return canonicalBoard.total_rew


        if self.Es[s] is not None:
            return self.Es[s]


        valids = game.getValidMoves(canonicalBoard)


        if sum(valids) == 0:
            return canonicalBoard.total_rew


        if s not in self.Ps:
            if self.args.MCTSmode == 0:
                v = canonicalBoard.total_rew
                self.Ps[s] = canonicalBoard.prob
            else:
                self.Ps[s], v = self.nnet.predict(canonicalBoard.get_state())


            self.Ps[s] = self.Ps[s] * valids


            if np.sum(self.Ps[s]) > 0:
                self.Ps[s] /= np.sum(self.Ps[s])
            else:
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])


            self.Ns[s] = 0
            return v


        cur_best = -float("inf")
        best_act = -1


        for a in range(game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    classical_ucb = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(
                        self.Ns[s]
                    ) / (1 + self.Nsa[(s, a)])
                else:
                    classical_ucb = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)


                gp_features = [
                    float(self.Ns.get(s, 0)),
                    float(self.Nsa.get((s, a), 0)),
                    float(self.Ps[s][a]),
                    float(a),
                ]


                gp_score = self.gp_helper.ucb(gp_features)
                alpha = self.args["gp_blend_alpha"] if "gp_blend_alpha" in self.args else 0.7


                u = alpha * classical_ucb + (1.0 - alpha) * gp_score
                self.GPsa[(s, a)] = gp_score


                if u > cur_best:
                    cur_best = u
                    best_act = a


        if best_act == -1:
            return canonicalBoard.total_rew


        a = best_act


        if (s, a) not in self.cache_data:
            game_copy_dir1 = game.get_copy()
            next_s_dir1 = game_copy_dir1.getNextState(canonicalBoard, a)


            comp_a = canonicalBoard.get_complement_action(a)
            game_copy_dir2 = game.get_copy()
            next_s_dir2 = game_copy_dir2.getNextState(canonicalBoard, comp_a)


            self.cache_data[(s, a)] = (next_s_dir1, canonicalBoard)
            self.cache_data[(s, comp_a)] = (next_s_dir2, canonicalBoard)
        else:
            comp_a = canonicalBoard.get_complement_action(a)
            next_s_dir1, canonicalBoard = self.cache_data[(s, a)]
            next_s_dir2, canonicalBoard = self.cache_data[(s, comp_a)]
            game_copy_dir1 = game.get_copy()
            game_copy_dir2 = game.get_copy()


        v1 = self.search(game_copy_dir1, next_s_dir1, verbose=verbose, level=level + 1)
        v2 = self.search(game_copy_dir2, next_s_dir2, verbose=verbose, level=level + 1)


        v = (v1 + v2) / 2 - self.args.varpen * abs(v1 - v2)


        gp_features = [
            float(self.Ns.get(s, 0)),
            float(self.Nsa.get((s, a), 0)),
            float(self.Ps[s][a]),
            float(a),
        ]
        self.gp_helper.add_observation(gp_features, float(v))


        if (s, a) in self.Qsa:
            self.Bsa[(s, a)] = max(self.Bsa[(s, a)], v)
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                self.Nsa[(s, a)] + 1
            )
            self.Nsa[(s, a)] += 1
        else:
            self.Bsa[(s, a)] = v
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1


        if (s, comp_a) in self.Qsa:
            self.Bsa[(s, comp_a)] = max(self.Bsa[(s, comp_a)], v)
            self.Qsa[(s, comp_a)] = (
                self.Nsa[(s, comp_a)] * self.Qsa[(s, comp_a)] + v
            ) / (self.Nsa[(s, comp_a)] + 1)
            self.Nsa[(s, comp_a)] += 1
        else:
            self.Bsa[(s, comp_a)] = v
            self.Qsa[(s, comp_a)] = v
            self.Nsa[(s, comp_a)] = 1


        self.Ns[s] += 1
        return v
