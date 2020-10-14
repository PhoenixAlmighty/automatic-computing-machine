from __future__ import print_function
import timeit, argparse, os, sys
import FFArbiterPUF
from mpi4py import MPI

import generator
import models as md
import utils


def get_args():
    parser = argparse.ArgumentParser(
        description="FF PUF Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stages', metavar="S", type=int,
                        default=64, help='Test size (0-1)')
    parser.add_argument('--challenges', metavar="C", type=int,
                        default=625000, help='Number of challenges')
    parser.add_argument('--num_loops', metavar="FF", type=int,
                        default=1, help='Number of FF loops')
    parser.add_argument('--overlap', metavar="FF", type=int,
                        default=True, help='Number of FF loops')
    parser.add_argument('--runs', metavar="R", type=int,
                        default=1, help='Number of runs')
    parser.add_argument('--solver', metavar="ChooseSolver", nargs='?',
                        default="adam", help='Optimization algorithm')
    parser.add_argument('--minibatch', metavar='minibatch_size', type=int,
                        default=200, help='Batch size for training')
    parser.add_argument('--num_layers', metavar='n_layers', type=int,
                        default=1, help='Number of layers in model')
    parser.add_argument('--num_neurons', metavar='n_neurons', type=int,
                        default=2, help='Number of neurons per layer if no FF loops')
    parser.add_argument('--max_proc', metavar="max_processes", type=int,
                        default=5, help='Maximum number of child processes')
    return parser.parse_args()


"""
--------------------------------------------------------------------------
                        Main Function
--------------------------------------------------------------------------
"""

if __name__ == "__main__":
    experiment_start_time = timeit.default_timer()
    args = get_args()
    
    filename = '../xor%d_(%d)bit.txt' % (args.num_loops, args.stages)
    output = open(filename, "a")
    model = md.PUF_MultilayerPerceptron(ff_loops=args.num_loops,
                                        num_stages=args.stages,
                                        minibatch=args.minibatch,
                                        solver=args.solver,
                                        n_layers=args.num_layers,
                                        n_neurons=args.num_neurons)

    # gen = generator.PUFGenerator(num_stages=args.stages,
                                 # ff_loops=args.num_loops,
                                 # overlaps=args.overlap)
    
    line = "\n-----------------------------------------------\n" \
           + '[{}] PUF: '.format('FF') \
           + 'FF_Loops={:d}, '.format(int(args.num_loops)) \
           + 'Stages={:d}bit, '.format(int(args.stages)) \
           + 'CRPs={:d}K\n'.format(int(args.challenges / 1000)) \
           + "-----------------------------------------------\n"
    print(line)
    output.write(line)

    FFArbiterPUF.FF_Breaker(output, args, model)
    elapsed = timeit.default_timer() - experiment_start_time
    time_, unit = utils.convert_time(float(elapsed))
    output.write('Experiment Total time= %.3f %s\n' % (time_, unit))
    print('Experiment Total time= %.3f %s\n\n' % (time_, unit))
    output.close()
