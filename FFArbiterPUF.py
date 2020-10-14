# General imports
from __future__ import print_function
import os, datetime, sys, time, itertools, math
import numpy as np
import generator
import breakerfunctions as bf

# Below code block is to avoid a certain type of crash
# By default, TensorFlow allocates most or all of available GPU
# memory to a process; when multiple TF processes are trying to
# use that GPU, this results in a program crash
# In Visual Studio Code (not tested with other IDEs),
# gpu_options is not recognized as a member of ConfigProto;
# however, code block still works as intended

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config = config))

import utils
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mpi4py import MPI


def FF_Breaker(output, args, model_nn):
    assert args.challenges % 1000 == 0
    k = args.challenges//1000
    assert k % args.max_proc == 0
    metrics = {m: [] for m in ['acc', 'time']}
    for run_index in range(args.runs):
        print('                                #########################')
        print('--------------------------------# Run[%d]   {%s} # ---------------------------------' % ((run_index + 1), datetime.datetime.now().strftime('%m-%d-%Y')))
        print('                                #########################')
        line = '\n\n-----------| RUN ' + str(run_index + 1) + ' |-----------\n'
        output.write(line)

        gen = generator.PUFGenerator(num_stages=args.stages,
                                 ff_loops=args.num_loops,
                                 overlaps=args.overlap)
        
        test_acc, train_time = train_test_nn(args, gen, model_nn)

        metrics['acc'].append(test_acc)
        metrics['time'].append(train_time)

        print('\t>> | RESULT |')
        line = '\t Test Acc = {:.2f}, '.format(test_acc) \
               + 'Train Time = {:.3f}, '.format(train_time) \
               + '[ AVG_ACC = {:.2f}, '.format(np.mean(metrics['acc'])) \
               + 'AVG_TIME = {:.3f} ]\n\n '.format(np.mean(metrics['time']))
        print(line)
        output.write(line)

def train_test_nn(args, gen, model):
    last_starting_loop_indx = sorted(list(gen.ff_loops_set.keys()))
    split = last_starting_loop_indx[-1]
    remaining = args.stages - split
    k = args.challenges//1000
    S1 = gen.generate_challenges(k, split)
    print('\n\t | ### %s ###| \n' % gen.ff_loops_set)
    
    comm = MPI.COMM_SELF.Spawn(sys.executable,
                               args = ['child.py'],
                               maxprocs = args.max_proc)
    recvbuf = None
    comm.Bcast(np.asarray([split, remaining, k], dtype = np.dtype(int)), root = MPI.ROOT)
    comm.Scatter(S1, recvbuf, root = MPI.ROOT)
    comm.bcast(gen, root = MPI.ROOT)
    comm.bcast(model, root = MPI.ROOT)
    
    length = np.empty(args.max_proc, dtype = np.int32)
    acc = np.empty(args.max_proc, dtype = np.float32)
    training_time = np.empty(1)
    # The code block starting at comm.Gather() and going on for 6
    # more lines is used instead of using comm.Gather() for the
    # results from the child processes because the Gather() method
    # requires the sizes of all send buffers to be equal, which is
    # not guaranteed in this use case
    comm.Gather(None, length, root = MPI.ROOT)
    CRP_set = []
    for i in range(0, args.max_proc):
        crps = np.empty((length[i], args.stages + 1), dtype = np.int8)
        comm.Recv(crps, source = i)
        CRP_set.extend(crps.tolist())
    CRP_set = np.asarray(CRP_set)
    comm.Reduce(None, [training_time, MPI.DOUBLE], op = MPI.MAX, root = MPI.ROOT)
    train_time = training_time[0]

    comm.Gather(None, acc, root = MPI.ROOT)
    best_acc = 0.0
    best_acc_index = 0
    for i in range(0, args.max_proc):
        if best_acc < acc[i]:
            best_acc = acc[i]
            best_acc_index = i
    best_or_not = np.zeros(args.max_proc, np.int8)
    best_or_not[best_acc_index] = 1
    comm.Scatter(best_or_not, None, root = MPI.ROOT)
    model = comm.recv(source = best_acc_index)

    comm.Disconnect()
    CRP_set = np.random.permutation(CRP_set)
    
    print('\n\n\t :: Final Model Training ::')
    print("\t>> Training set size: %s" % str(CRP_set.shape))
    tr_r = CRP_set[:, -1]
    tr_l = utils.responses_to_labels(tr_r)
    C = np.delete(CRP_set, -1, axis=1)
    tr_C = utils.transformation(C)

    start = time.time()
    model.fit(tr_C, tr_l, ep = 1)
    end = time.time()
    train_time = train_time + (end - start)

    test_set = bf.handle_testing_dataset(gen, args, CRP_set)
    print("\t>> Testing set size: %s" % str(test_set.shape))
    ts_r = test_set[:, -1]
    C = np.delete(test_set, -1, axis=1)
    ts_C = utils.transformation(C)
    test_pred = model.predict(ts_C)
    test_pred = utils.labels_to_responses(test_pred)
    test_acc = accuracy_score(ts_r, test_pred) * 100.

    return test_acc, train_time
