import numpy as np
import time
from adt.models.dqn_agent import DQNagent
from adt.envs.dqn_env import Env
import os


def train_dqn(dataset, episodes, batch_size, l_action, k_state):
    state_size = 6
    action_size = 2
    action_set = [0, 1]
    target_update = 100
    rewards_list = []
    action_hist = []
    score_path = os.path.join("processed_data", dataset, "ae_score.npy")
    labels_path = os.path.join("processed_data", dataset, "windows_attack_labels.npy")
    # specify training data for different datasets based on their specific characteristics, make sure it includes both normal and abnormal samples. The training set can be flexibly adjusted.
    if dataset == "SWaT":
        score_ae = np.load(score_path)[1500:2000]
        y_labels = np.load(labels_path)[1500:2000]
    elif dataset == "WADI":
        score_ae = np.load(score_path)[59000:60000]
        y_labels = np.load(labels_path)[59000:60000]
    elif dataset=="Yahoo":
        score_ae = np.load(score_path)[25905: 26905]
        y_labels = np.load(labels_path)[25905: 26905]
    elif dataset == "HAI":
        indexes = [(2050, 2150), (2250, 2350), (8700, 8900)]
        score_ae = np.load(score_path)
        y_labels = np.load(labels_path)
        score_ae = np.concatenate([score_ae[start:end] for start, end in indexes])
        y_labels = np.concatenate([y_labels[start:end] for start, end in indexes])
    print("Score range: ", min(score_ae), max(score_ae))
    print("Training size:", len(score_ae), "Anomaly count:", sum(y_labels),
          "Max score:", np.max(score_ae), "Min score:", np.min(score_ae))

    agent = DQNagent(state_size, action_size)
    env = Env(action_set, y_labels, score_ae, verbose=True)
    epsilon = 1
    epsilon_min = 0.001
    epsilon_list = [epsilon]
    start_time = time.time()
    for e in range(episodes):
        state = env.reset()
        e_reward = 0
        for t_env in range(len(y_labels)):
            if t_env % l_action == 0:
                action_index = agent.policy(state, epsilon)
                action = action_set[action_index]
                action_hist.append(action)
            else:
                action = action_hist[-1]
            reward, next_state, done = env.do_step(action, t_env, k_state, e, episodes)
            e_reward += reward
            state = next_state
            agent.replaymemory.store_experiences(state, action_set.index(action), reward, next_state, done)
            if t_env == len(y_labels) - 1 and len(agent.replaymemory.memory) >= batch_size:
                mini_batch = agent.replaymemory.sample_memory(batch_size)
                agent.train(mini_batch, e, episodes)
                epsilon = max(epsilon - 1 / (episodes * 0.99), epsilon_min)
                epsilon_list.append(epsilon)
        rewards_list.append(e_reward)
        if e % target_update == 0:
            agent.update_target_net()
        if e == episodes - 1 or e % 200 == 0:
            print("Episode {}/{}: reward = {}, epsilon = {}".format(e + 1, episodes, e_reward, epsilon))
        if e == episodes - 1:
            print("Training time:", time.time() - start_time)
            model_save_path = os.path.join("saved_models", dataset, "dqn_model.h5")
            agent.q_net.save(model_save_path)
            print("Trained DQN model saved at:", model_save_path)


if __name__ == '__main__':
    train_dqn(episodes=2000, batch_size=256, l_action=10, k_state=1)
