import logging
import time
import os

import coloredlogs

from Coach import Coach
from ksgraph.KSGame import KSGame
from utils import *

import wandb
import argparse
import random
import numpy as np

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')

random.seed(42)
np.random.seed(42)

args_f = {
    'numIters': 1,
    'numEps': 1,
    'updateThreshold': None,
    'maxlenOfQueue': 200000,
    'arenaCompare': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'CCenv': True,
    'model_name': 'MCTS',
    'model_notes': 'MCTS without NN',
    'model_mode': 'mode-0',
    'phase': 'scaling',
    'version': 'v1',

    'debugging': True,
    'verbose': True,
    'wandb_logging': False,

    'MCTSmode': 0,
    'nn_iter_threshold': 5,

    'STATE_SIZE': 10,
    'STEP_UPPER_BOUND_MCTS': 20,

    'LIMIT_TOP_3': True,

    # GP-UCB defaults
    'gp_beta': 2.0,
    'gp_blend_alpha': 0.7,
}


def main(args_parsed):
    args = dotdict({**args_f, **vars(args_parsed)})

    if args.prod:
        args['debugging'] = False
        args['verbose'] = False

    if args.d != -1:
        args['STEP_UPPER_BOUND'] = args_parsed.d
    else:
        args['STEP_UPPER_BOUND'] = None

    if args.n != -1:
        args['VARS_TO_ELIM'] = args_parsed.n
    else:
        args['VARS_TO_ELIM'] = None

    args['MAX_LITERALS'] = args_parsed.m

    if args.debugging:
        print(args)

    if not args.wandb_logging:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            reinit=True,
            name=args.version + "_" + args.model_name + "_" + args.model_mode + "_" + args.phase,
            project="AlphaSAT",
            tags=[args.model_name, args.model_mode, args.phase, args.version],
            notes=args.model_notes,
            settings=wandb.Settings(start_method='fork' if args.CCenv else 'thread'),
            save_code=True
        )

    wandb.config.update(args)

    g = KSGame(args=args, filename=args.filename)
    nnet = None

    if args.debugging:
        log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.MCTSmode == 0:
        c.nolearnMCTS()
        return


if __name__ == "__main__":
    start_time_tool = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="filename of the CNF file", type=str)
    parser.add_argument("-n", help="cutoff when n variables are eliminated", type=int, default=-1)
    parser.add_argument("-d", help="cutoff when d depth is reached", type=int, default=-1)
    parser.add_argument("-m", help="only top m variables to be considered for cubing", type=int)
    parser.add_argument("-o", help="output file for cubes")

    parser.add_argument("-cpuct", help="cpuct for MCTS", type=float, default=10)
    parser.add_argument("-numMCTSSims", help="MCTS sims", type=int, default=10)
    parser.add_argument("-varpen", help="Variance penalty factor", type=float, default=0)
    parser.add_argument("-nMCTSEndOfG", help="MCTS end of game criteria (n cutoff)", type=int, default=-1)

    # Optional GP-UCB overrides from command line
    parser.add_argument("-gp_beta", help="GP-UCB beta value", type=float, default=2.0)
    parser.add_argument("-gp_blend_alpha", help="Blend weight for classical MCTS score", type=float, default=0.7)

    parser.add_argument("-prod", action="store_true", help="production mode")

    args_parsed = parser.parse_args()
    if not args_parsed.prod:
        print(args_parsed)

    args = parser.parse_args()

    if args_parsed.n == -1 and args_parsed.d == -1:
        print("Either -n or -d must be specified")
        exit()

    if not os.path.exists(f"cubing_outputs/{args.o}"):
        os.makedirs(f"cubing_outputs/{args.o}")

    main(args_parsed)

    time_elapsed_tool = time.time() - start_time_tool
    print("Tool runtime: ", round(time_elapsed_tool, 3))