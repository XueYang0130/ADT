import random
import tensorflow as tf
from keras import Sequential
from tensorflow import keras
from keras.layers import Input, Dense
from collections import deque
import numpy as np


class ReplayMemory():
    def __init__(self):
        self.memory = deque(maxlen=20000)

    def store_experiences(self, state, action_index, reward, next_state, done):
        self.memory.append([state, action_index, reward, next_state, done])

    # randomly sample a batch of experiences for training purpose
    def sample_memory(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        state_batch, action_index_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []
        for i in minibatch:
            state_batch.append(i[0])
            # action_batch.append(i[1])
            action_index_batch.append(i[1])
            reward_batch.append(i[2])
            next_state_batch.append(i[3])
            done_batch.append(i[4])
        return state_batch, action_index_batch, reward_batch, next_state_batch, done_batch


class DQNagent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.replaymemory = ReplayMemory()
        self.learning_rate = 1e-4
        self.action_set=[0, 1]

        self.q_net = self._build_model()
        self.target_net = self._build_model()

    # creat deep neural network
    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        return model

    # define the policy given a state - how the agent acts by exploration and exploitation
    # generate the action index
    def policy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        else:
            action_q = self.q_net(np.array([state])).numpy()
            return np.argmax(action_q)

    # train q_net
    def train(self, batch, e, Episodes):
        state_batch, action_index_batch, reward_batch, next_state_batch, done_batch = batch
        current_q_batch = self.q_net(np.array(state_batch)).numpy()
        target_q_batch = np.copy(current_q_batch)
        next_q_batch = self.target_net(np.array(next_state_batch)).numpy()
        max_next_q_batch = np.amax(next_q_batch, axis=1)  # np.amax(...,axis=1): max of each row
        for i in range(len(target_q_batch)):
            target_q_batch[i][action_index_batch[i]] = reward_batch[i] if done_batch[i] else reward_batch[i] + self.gamma * \
                                                                                       max_next_q_batch[i]
        # train q_net
        result = self.q_net.fit(x=tf.convert_to_tensor(state_batch), y=tf.convert_to_tensor(target_q_batch), verbose=0)
        loss = result.history['loss']

        # # save model
        # if e == Episodes - 1:
        #     self.q_net.save("saved_models/dqn_model.h5")

    # assign the weights of q_net periodically to update the target net weights (
    def update_target_net(self):
        self.target_net.set_weights(self.q_net.get_weights())
