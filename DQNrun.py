from DQNAgent import DQNagent
from DQNEnv import Env
import numpy as np
import time
import keras


def sliding_window(target, labels, window_size=10, stripe=1):
    x = []
    y = []
    for i in range(0, len(target) - window_size + 1, stripe):
        x.append(target[i:i + window_size])
        y.append(labels[i:i + window_size])
    return np.array(x), np.array(y)

def eval_performance(tp_n, fp_n, fn_n):
    if tp_n == fp_n == fn_n == 0:
        precision = recall = f1 = 1
    elif tp_n == 0 and np.sum(fp_n + fn_n) > 0:
        precision = recall = f1 = 0
    else:
        precision = tp_n / (tp_n + fp_n)
        recall = tp_n / (tp_n + fn_n)
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def train(train_flag, batch_size, l_action, k_state, Episodes):
    # initialization
    state_size = 6
    action_size = 2
    action_set = [0, 1]
    target_update = 100
    rewards_list = []
    action_hist = []

    # load SWAT dataset
    if train_flag:
        score_ae = np.load("datasets//ae_score.npy")[1500:2000]
        y_labels = np.load("datasets//windows_attack_labels.npy")[1500:2000]
        print("training size:", len(score_ae), "anomaly count:", sum(y_labels), "max score:", np.max(score_ae),
              "min score:", np.min(score_ae))
        # create an agent and  environment
        agent = DQNagent(state_size, action_size)
        env = Env(action_set, y_labels, score_ae)
        epsilon = 1
        epsilon_min = 0.001
        epsilon_list = [epsilon]

    if not train_flag:
        score_ae = np.load("datasets//ae_score.npy")
        y_labels = np.load("datasets//windows_attack_labels.npy")
        print("training size:", len(score_ae), "anomaly count:", sum(y_labels), "max score:", np.max(score_ae),
              "min score:", np.min(score_ae))
        # load the trained model
        agent = DQNagent(state_size, action_size)
        env = Env(action_set, y_labels, score_ae)
        agent.q_net = agent.target_net = keras.models.load_model("model")
        epsilon = 0

    for e in range(0, Episodes):
        state = env.reset()
        e_reward = 0
        for t_env in range(len(y_labels)):
            if t_env % l_action == 0:
                action_index = agent.policy(state, epsilon)
                action = action_set[action_index]
                action_hist.append(action)
            else:
                action = action_hist[-1]

            reward, next_state, done = env.do_step(action, t_env, k_state, e, Episodes)
            e_reward += reward
            state = next_state
            if train_flag:
                agent.replaymemory.store_experiences(state, action_set.index(action), reward, next_state, done)
            # if training, update Q network
            if train_flag and t_env == len(y_labels) - 1:
                if len(agent.replaymemory.memory) >= batch_size:
                    mini_batch = agent.replaymemory.sample_memory(batch_size)
                    agent.train(mini_batch, e, Episodes)
                    epsilon = max(epsilon - 1 / (Episodes * 0.99), epsilon_min)
                    epsilon_list.append(epsilon)
        rewards_list.append(e_reward)
        if train_flag:  # if training, update target network
            if e % target_update == 0:
                agent.update_target_net()
            if e == Episodes - 1 or e % 200 == 0:
                print("episode:{}/{}, reward:{}, epsilon:{}".format(e + 1, Episodes, e_reward, epsilon))
        if not train_flag:  # testing result
            precision, recall, f1_score = eval_performance(env.tp_n, env.fp_n, env.fn_n)
            print("Testing Result==> precision:{} recall:{} F1: {}".format(precision, recall, f1_score))
            print("TP:{} TN:{} FP:{} FN:{}".format(env.tp_n, env.tn_n, env.fp_n, env.fn_n))

if __name__ == '__main__':
    # train
    train(train_flag=True, batch_size=256, l_action=10, k_state=10, Episodes=2000)

    # test
    #train(train_flag=False, batch_size=256, l_action=1, k_state=10, Episodes=1)
