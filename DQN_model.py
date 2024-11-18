from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Activation, Flatten , InputLayer
from keras.optimizers import Adam
import numpy as np
from collections import deque
import time
from modified_tensor_board import ModifiedTensorBoard
from keras.callbacks import TensorBoard
from constants import *
import random


class DQN_model:
    def __init__(self):
        self.replay_memory_size = 50_000
        self.min_replay_mem_size = 1_000

        self.batch_size = 64
        self.input_shape = 10
        self.output_shape = 5

        self.discount_factor = 0.99

        self.MODEL_NAME = "DQN_pacman_model"

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=self.replay_memory_size)
        self.tensorboard = ModifiedTensorBoard(
            log_dir="logs/fit/{}-{}".format(self.MODEL_NAME, int(time.time()))
        )

        self.target_update_ctr = 0
        self.target_update_period = 5

    def create_model(self):
        model = Sequential()

        model.add(InputLayer(shape=(self.input_shape,)))

        model.add(Dense(units=64, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(units=32, activation="relu"))
        # model.add(Dropout(0.2))

        model.add(Dense(units=16, activation="relu"))
        # model.add(Dropout(0.2))

        # model.add(Dense(units=16, activation="relu"))
        # model.add(Dropout(0.2))

        # model.add(Dense(units=8, activation="relu"))
        # model.add(Dropout(0.2))

        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(units=self.output_shape, activation="linear"))
        model.compile(
            loss="mse", optimizer=Adam(learning_rate=0.01)  ########### we can enhance the learning rate a little bit and see hat happens make it .01
        )
        return model

    def update_replay_memory(self, experience):
        self.replay_memory.append(experience)

    def get_qs(self, state):
        return (self.model.predict(state.reshape(-1, *state.shape)))[0]

    def train(self):
        if len(self.replay_memory) < self.min_replay_mem_size:
            return

        exps_batch = random.sample(self.replay_memory, self.batch_size)

        current_states = np.array([exp[0] for exp in exps_batch])
        current_qs = self.model.predict(current_states)

        future_states = np.array([exp[2] for exp in exps_batch])
        future_qs = self.target_model.predict(future_states)

        x = []
        y = []

        for index, (curr_state, action, new_state, reward, done) in enumerate(
            exps_batch
        ):
            if not done:
                max_future_q = np.max(future_qs[index])
                real_q = reward + self.discount_factor * max_future_q
            else:
                real_q = reward

            real_qs = current_qs[index]
            real_qs[action] = real_q

            x.append(curr_state)
            y.append(real_qs)

        self.model.fit(
            np.array(x),
            np.array(y),
            batch_size=self.batch_size,
            callbacks=[self.tensorboard],
            verbose=False,
            shuffle=False,
        )

        self.target_update_ctr += 1

        if self.target_update_ctr >= self.target_update_period:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_ctr = 0
