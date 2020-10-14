from mpi4py import MPI

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

import itertools
import numpy as np
import models as md
import breakerfunctions as bf
import generator
import utils


comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

sr = np.empty(3, dtype = np.dtype(int))
gen = None
model = None

comm.Bcast(sr, root = 0)

S1 = np.empty(((sr[2]//size), sr[0]), dtype = np.int8)

comm.Scatter(None, S1, root = 0)
gen = comm.bcast(gen, root = 0)
model = comm.bcast(model, root = 0)

train_time = 0.0
best_acc = 0.0
CRP_set = []
end_loop_indx = sorted(list(gen.ff_loops_set.values()))

for id_, i in enumerate(S1):
    SV1 = []
    S2 = gen.generate_challenges(1000, sr[1])
    for j in S2:
        v = np.concatenate([i, j])
        r = gen.simulate_one_challenge(v)
        v = np.concatenate([v, [r]])
        SV1.append(v)
        del v, r

    batch_acc, train_time, best_W, b_, SV1_best = bf.find_best_batch(gen, model, train_time, end_loop_indx, SV1)
    if batch_acc > best_acc:
        best_acc = batch_acc
        model.weights = best_W
    CRP_set.extend(SV1_best)
    
    print('\t\t>> S%s=%s set to (%s) possible batches. The best one accuracy= %.2f%%' % (id_ + 1, b_, 2 ** gen.ff_loops, batch_acc))
    # if SV1_best != None:
        # CRP_set.extend(SV1_best)
    del SV1_best, SV1
    
CRP_set = np.asarray(CRP_set, dtype = np.int8)
CRP_length = np.asarray([CRP_set.shape[0]], dtype = np.int32)
training_time = np.asarray([train_time])
comm.Gather(CRP_length, None, root = 0)
comm.Send(CRP_set, dest = 0)
comm.Reduce([training_time, MPI.DOUBLE], None, op = MPI.MAX, root = 0)

acc = np.asarray([best_acc], dtype = np.float32)
comm.Gather(acc, None, root = 0)
best_or_not = np.empty(1, dtype = np.int8)
comm.Scatter(None, best_or_not, root = 0)
if best_or_not[0] == 1:
    comm.send(model, dest = 0)
# add in figuring out whether this one is supposed to send its model to the parent

comm.Disconnect()
