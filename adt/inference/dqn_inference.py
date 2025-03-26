import numpy as np
import time
import keras
from adt.models.dqn_agent import DQNagent
from adt.envs.dqn_env import Env
import os


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


def test_dqn(model_dataset, test_dataset, l_action=10, k_state=1):
    state_size = 6
    action_size = 2
    action_set = [0, 1]
    action_hist = []
    score_path = os.path.join("processed_data", test_dataset, "ae_score.npy")
    labels_path = os.path.join("processed_data", test_dataset, "windows_attack_labels.npy")
    score_ae = np.load(score_path)
    y_labels = np.load(labels_path)
    print("Score range: ", min(score_ae), max(score_ae))
    print("Training size:", len(score_ae), "Anomaly count:", sum(y_labels),
          "Max score:", np.max(score_ae), "Min score:", np.min(score_ae))

    agent = DQNagent(state_size, action_size)
    env = Env(action_set, y_labels, score_ae, verbose=False)
    dqn_model_path = os.path.join("saved_models", model_dataset, "dqn_model.h5")
    agent.q_net = agent.target_net = keras.models.load_model(dqn_model_path)
    epsilon = 0
    start_time = time.time()
    for t_env in range(len(y_labels)):
        if t_env % l_action == 0:
            action_index = agent.policy(env.state, epsilon)
            action = action_set[action_index]
            action_hist.append(action)
        else:
            action = action_hist[-1]
        reward, next_state, done = env.do_step(action, t_env, k_state=1, e=0, Episodes=1)
        env.state = next_state
        if done:
            break
    precision, recall, f1_score = eval_performance(env.tp_n, env.fp_n, env.fn_n)
    print(f"{test_dataset} Inference using model trained on {model_dataset} ==> Precision: {precision}, Recall: {recall}, F1: {f1_score}, Testing time: {time.time() - start_time} seconds")
    print("TP: {}, TN: {}, FP: {}, FN: {}".format(env.tp_n, env.tn_n, env.fp_n, env.fn_n))


if __name__ == '__main__':
    test_dqn(l_action=1, k_state=1)
