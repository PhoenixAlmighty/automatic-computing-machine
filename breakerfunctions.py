# General imports
from __future__ import print_function
import os, datetime, sys, time, itertools, math
import numpy as np
import generator # , features
import models as md

import utils
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def handle_testing_dataset(gen, args, CRP_set):

    # Generate new testing dataset
    S = gen.generate_challenges(int(args.challenges * 0.3), args.stages)
    ts_C, ts_r = gen.simulate_challenges(S)
    S_test = np.column_stack((ts_C, ts_r))

    # Reed's note: I'm not sure what the purpose of the code from here to
    # the return statement is -- I can see that it changes some of the test
    # set, but I don't understand why. Is it to introduce some noise to the
    # test set to see how well the model handles it?

    # Starting from leftmost end_ff_loop
    end_ff_loops = sorted(list(gen.ff_loops_set.values()), reverse=False)
    for i in end_ff_loops:
        starting_ff_loop = utils.getKeysByValue(gen.ff_loops_set, i)[0]
        tr_C = []
        tr_r = []
        for j in CRP_set:
            tr_r.append(j[i])
            tr_C.append(j[:starting_ff_loop+1])
        tr_r = np.asarray(tr_r)
        tr_C = np.asarray(tr_C)
        temp = np.column_stack((tr_C, tr_r))
        print(temp.shape)
        temp = np.unique(temp, axis=0)
        print(temp.shape)
        tr_r = temp[:, -1]
        tr_C = np.delete(temp, -1, axis=1)

        sub_model = md.PUF_MultilayerPerceptron(n_neurons = 2, num_stages = temp.shape[1] - 1)
        tr_C = utils.transformation(tr_C)
        tr_l = utils.responses_to_labels(tr_r)
        sub_model.fit(tr_C, tr_l)

        temp = []
        for j in S_test:
            temp.append(j[:starting_ff_loop+1])
        pred_r = utils.labels_to_responses(sub_model.predict(np.asarray(temp)))

        for idx, j in enumerate(S_test):
            S_test[idx][i] = pred_r[idx]
    return S_test

def find_best_batch(gen, model, train_time, end_loop_indx, SV1):
    p = np.random.permutation(np.array(list(itertools.product([0, 1], repeat=gen.ff_loops))))
    batch_acc = 0.0
    SV1_best = None
    b_ = None
    best_W = None

    for id in range(2 ** gen.ff_loops):
        batch = np.copy(SV1)
        for c in batch:
            for b in range(gen.ff_loops):
                c[end_loop_indx[b]] = p[id][b]
        r = batch[:, -1]
        C = np.delete(batch, -1, axis=1)
        test_acc, train_time1 = model_training(C, r, model)
        train_time = train_time + train_time1

        if test_acc >= batch_acc:
            batch_acc = test_acc
            SV1_best = batch.tolist()
            best_W = model.estimator.get_weights()
            b_ = batch.shape
    return batch_acc, train_time, best_W, b_, SV1_best


def model_training(C, r, model):
    C = utils.transformation(C)
    tr_C, ts_C, tr_r, ts_r = train_test_split(C, r, train_size=.8)
    tr_l = utils.responses_to_labels(tr_r)
    start = time.time()
    model.fit(tr_C, tr_l, ep = 1)
    end = time.time()

    test_pred = model.predict(ts_C)
    test_pred = utils.labels_to_responses(test_pred)
    test_acc = accuracy_score(ts_r, test_pred) * 100.

    return test_acc, (end - start)
