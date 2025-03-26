## ADT: Agent-based Dynamic Thresholding for Time Series Anomaly Detection

ADT is an adaptive anomaly detection system that leverages AutoEncoder (AE) and Deep Q-Network (DQN) to detect anomalies in time series. It can either be used as a dynamic thresholding controller or directly as an anomaly detector.
The project supports multiple benchmark datasets (SWaT, WADI, HAI, Yahoo, etc.).

> **Note:**  
> - Raw datasets (under `dataset/`) are not included due to copyright restrictions.  
> - Dataset versions used: SWaT 2015, WADI 2017, HAI 21.03, Yahoo A1Benchmark. Please refer to the official sites.
> - Preprocessed data (`processed_data/`) and trained models (`saved_models/`) are excluded via `.gitignore` to reduce repository size.

---

### Project Structure
```
ADT/
├── adt/                  
│   ├── data/             # Data preprocessing
│   ├── envs/             # DQN environment
│   ├── inference/        # DQN inference scripts
│   ├── models/           # Model definitions (AE, DQN)
│   └── training/         # DQN training
├── examples/             # Pipeline entry point
│   └── run_pipeline.py
├── processed_data/       # Preprocessed datasets (not tracked)
│   ├── SWaT/
│   ├── WADI/
│   ├── HAI/
│   └── Yahoo/
├── saved_models/         # Trained models (not tracked)
│   ├── SWaT/
│   ├── WADI/
│   ├── HAI/
│   └── Yahoo/
├── dataset/              # Raw data (excluded)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

```

### Features

- **Data Preprocessing:**  
  Preprocess raw datasets to create sliding windows, labels, and normalized data stored in `processed_data/`.  
- **AE Training:**  
  Train an AutoEncoder model on the preprocessed data and generate anomaly scores. The trained AE model is saved to `saved_models/<dataset>/ae_model.h5`.
- **DQN Training:**  
  Train a DQN model using the anomaly scores from the AE, with the trained model saved to `saved_models/<dataset>/dqn_model.h5`.
- **Inference:**  
  Load the pretrained DQN model and perform inference on the processed data, reporting performance metrics (Precision, Recall, F1 Score).
- **Robust Testing (Optional):**  
  Test the robustness of a DQN model trained on one dataset on another dataset by specifying a separate model dataset.

---

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/XueYang0130/ADT.git
   cd ADT
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Note on Raw Data**
   ```bash
   Raw datasets should be placed in the dataset/ folder.

### Usage

1. **Data Preparation:**  
To preprocess a specific dataset (if you have your own raw data), run:

```bash
python -m examples.run_pipeline --dataset=SWaT --task=prepare_data
```

> Note: change `SWaT` to any other dataset that you work on.

---

2. **Train the AE Model:**  
To train the AutoEncoder model on a specific dataset (e.g., SWaT):

```bash
python -m examples.run_pipeline --dataset=SWaT --task=run_ae
```

---

3. **Train the DQN Model:**  
To train the DQN model on a specific dataset (default `l_action=10`, `k_state=1`):

```bash
python -m examples.run_pipeline --dataset=SWaT --task=train_dqn
```

This command trains the AE model using data from `processed_data/SWaT/`,  
saves the model in `saved_models/SWaT/ae_model.h5`,  
and stores the anomaly scores in `processed_data/SWaT/ae_score.npy`.

---

4. **Inference:**  
To run inference using the trained DQN model (default `l_action=1`, `k_state=1`):

```bash
python -m examples.run_pipeline --dataset=SWaT --task=dqn_inference 
```

This loads the model from `saved_models/SWaT/dqn_model.h5`  
and performs inference on the processed data in `processed_data/SWaT/`,  
outputting performance metrics.

---

5. **Robust Testing (Optional):**  
To test the robustness of a model trained on one dataset with another dataset’s test data, use the optional `--model_dataset` argument:

```bash
python -m examples.run_pipeline --dataset=WADI --task=dqn_inference --model_dataset=SWaT
```

This command uses the DQN model trained on SWaT  
(from `saved_models/SWaT/dqn_model.h5`)  
to perform inference on the WADI dataset (from `processed_data/WADI/`).

---

### Key Results:

- **ADT significantly enhances anomaly detection performance**, achieving **F1 scores close to 1** on multiple benchmark datasets.
- It exhibits **strong robustness**, maintaining high accuracy even when trained on one dataset and applied to another.
- Interestingly, **ADT does not rely heavily on the quality of anomaly scores**, as it adapts effectively regardless of score precision.
- As a **change point detector**, its performance is **highly influenced by the underlying data structure**.
- It requires **only a minimal amount of labeled data**, making it practical for real-world scenarios where labeled anomalies are scarce.

---

**For detailed results, please refer to the following publication:**  
Yang, Xue, Enda Howley, and Michael Schukat.  
*"Agent-based dynamic thresholding for adaptive anomaly detection using reinforcement learning."*  
Neural Computing and Applications (2024): 1–17.  
https://link.springer.com/article/10.1007/s00521-024-10536-0

---

Furthermore, we validate ADT's robustness in more challenging scenarios, including environments with **noisy, partial, or delayed feedback**.  
Please refer to the following publication:  
Yang, Xue, Enda Howley, and Michael Schukat.  
*"ADT: Time series anomaly detection for cyber-physical systems via deep reinforcement learning."*  
Computers & Security 141 (2024): 103825.  
https://www.sciencedirect.com/science/article/pii/S0167404824001263
