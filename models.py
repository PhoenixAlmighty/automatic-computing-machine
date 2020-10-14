# This import line may not be necessary; we'll see
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import utils


class PUF_MultilayerPerceptron():
    train_time = 0.0

    def __init__(self, ff_loops = 1, num_stages = 64, minibatch = 200, solver = 'adam', n_layers = 1, n_neurons = 16):
        self.ff_loops = ff_loops
        self.num_stages = num_stages
        self.batch_size = minibatch
        self.weights = None
        self.solver = solver
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.build_model()
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(units = (self.n_neurons ** (self.ff_loops+1)), activation = 'relu', input_shape = (self.num_stages,)))
        for _ in range(1, self.n_layers):
            model.add(Dense(units = (self.n_neurons ** (self.ff_loops+1)), activation = 'relu'))
        model.add(Dense(units = 2, activation = 'softmax'))
        model.compile(optimizer = self.solver,
                      loss = 'categorical_crossentropy',
                      metrics = ['accuracy']
                      )
        self.estimator = model
    
    def fit(self, C, r, ep = 1):
        C, r = np.array(C), np.array(r)
        early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, patience = 10)
        if self.weights != None:
            self.estimator.set_weights(self.weights)
        return self.estimator.fit(x = C,
                                  y = r,
                                  batch_size = self.batch_size,
                                  epochs = ep,
                                  verbose = 0,
                                  callbacks = [early_stop],
                                  validation_split = 0.1
                                  )

    def predict(self, C):
        C = np.array(C)
        return self.estimator.predict(x = C)
