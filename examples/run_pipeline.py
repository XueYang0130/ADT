from adt.data.load_data import prepare_swat, prepare_wadi, prepare_hai, prepare_yahoo
from adt.models.ae import run_ae
from adt.training.dqn_training import train_dqn
from adt.inference.dqn_inference import test_dqn
import argparse
import time
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Unified Pipeline for ADT Project")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["SWaT", "WADI", "HAI","Yahoo"],
                        help="Select the dataset to process.")
    parser.add_argument("--task", type=str, required=True,
                        choices=["prepare_data", "run_ae", "train_dqn", "dqn_inference"],
                        help="Select the task to execute: prepare_data, train_ae, train_dqn, or inference.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for DQN training (if applicable).")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Number of episodes for DQN training (if applicable).")
    parser.add_argument("--l_action", type=int, default=10,
                        help="Control action update frequency for DQN training/inference.")
    parser.add_argument("--k_state", type=int, default=1,
                        help="State window size for DQN training/inference.")
    parser.add_argument("--model_dataset", type=str,
                        choices=["SWaT", "WADI", "HAI", "Yahoo"],
                        help="If specified, use this dataset's trained DQN model for inference.")
    args = parser.parse_args()

    start_time = time.time()

    if args.dataset == "SWaT":
        preprocess_func = prepare_swat
        data_dir = os.path.join("processed_data", "SWaT")
    elif args.dataset == "WADI":
        preprocess_func = prepare_wadi
        data_dir = os.path.join("processed_data", "WADI")
    elif args.dataset=="Yahoo":
        preprocess_func = prepare_yahoo
        data_dir = os.path.join("processed_data", "Yahoo")
    elif args.dataset == "HAI":
        preprocess_func = prepare_hai
        data_dir = os.path.join("processed_data", "HAI")
    else:
        parser.error("Unknown dataset specified.")

    if args.task == "prepare_data":
        print("Starting data preparation for", args.dataset)
        preprocess_func()
        print("Data preparation finished.")
    elif args.task == "run_ae":
        print("Starting AE training for", args.dataset)
        trained_ae, scores = run_ae(dataset=args.dataset)
        print("AE training finished and anomaly scores saved.")
    elif args.task == "train_dqn":
        print("Starting DQN training...")
        train_dqn(dataset=args.dataset, episodes=args.episodes, batch_size=args.batch_size, l_action=args.l_action, k_state=args.k_state)
        print("DQN training finished.")
    elif args.task == "dqn_inference":
        model_dataset = args.model_dataset if args.model_dataset else args.dataset
        print(f"Starting inference: using model trained on {model_dataset} to test on {args.dataset}")
        test_dqn(model_dataset=model_dataset, test_dataset=args.dataset, l_action=args.l_action, k_state=args.k_state)
        print("Inference finished.")
    else:
        parser.error("Unknown task specified.")
    elapsed = time.time() - start_time
    print("Total execution time: %.2f seconds", elapsed)

if __name__ == '__main__':
    main()
