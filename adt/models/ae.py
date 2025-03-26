import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import time
import matplotlib.pyplot as plt


def build_ae(window_dim, latent_dim):
    original_inputs = tf.keras.Input(shape=(window_dim,), name="encoder_input")
    x = layers.Dense(int(window_dim / 2), activation="relu")(original_inputs)
    x = layers.Dense(int(window_dim / 4), activation="relu")(x)
    z = layers.Dense(latent_dim, name="z")(x)
    AE_encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")

    latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z")
    x = layers.Dense(int(window_dim / 4), activation="relu")(latent_inputs)
    x = layers.Dense(int(window_dim / 2), activation="relu")(x)
    outputs = layers.Dense(window_dim, activation="sigmoid")(x)
    AE_decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

    outputs = AE_decoder(z)
    ae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="auto-encoder")
    return ae


def train_ae(ae, x_train, epochs=30, batch_size=8000, callbacks=None):
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
               loss=tf.keras.losses.MeanSquaredError())
    history = ae.fit(x=x_train, y=x_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    return history, ae


def generate_scores(ae, x_all):
    x_pred = ae(x_all).numpy()
    loss_fn = tf.keras.losses.MeanSquaredError()
    score_list = [loss_fn(x_all[i], x_pred[i]).numpy() for i in range(x_all.shape[0])]
    return np.array(score_list)


def run_ae(dataset, epochs=30, batch_size=8000, plot_result=True):
    data_dir = os.path.join("processed_data", dataset)
    normal_path = os.path.join(data_dir, "windows_normal_flatten.npy")
    attack_path = os.path.join(data_dir, "windows_attack_flatten.npy")
    x_window_normal = np.load(normal_path)
    x_all = np.load(attack_path)
    window_size = 12
    if dataset == "SWaT":
        input_size = 51
    elif dataset == "WADI":
        input_size = 123
    elif dataset == "HAI":
        input_size = 57
    elif dataset == "Yahoo":
        input_size = 1
    else:
        raise ValueError("Unknown dataset:" + dataset)
    window_dim = window_size * input_size
    latent_dim = 100
    if dataset == "Yahoo":
        latent_dim = 3
        epochs = 100
    ae = build_ae(window_dim, latent_dim)
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.9, patience=20, min_lr=1e-20)]
    start_time = time.time()
    history, trained_ae = train_ae(ae, x_window_normal[:int(len(x_window_normal) * 0.7)], epochs=epochs,
                                   batch_size=batch_size, callbacks=callbacks)
    training_time = time.time() - start_time
    print("AE Training time: {:.2f} seconds".format(training_time))
    scores = generate_scores(trained_ae, x_all)
    ae_score_path = os.path.join(data_dir, "ae_score.npy")
    np.save(ae_score_path, scores)
    print(f"ae_score.npy has been saved in {ae_score_path}")
    if plot_result:
        plt.plot(history.history["loss"])
        plt.suptitle(f'{dataset} AE Training Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    model_save_dir = os.path.join("saved_models", dataset)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = os.path.join(model_save_dir, "ae_model.h5")
    trained_ae.save(model_save_path)
    print(f"Trained AE model saved at: {model_save_path}")
    return trained_ae, scores


if __name__ == '__main__':
    run_ae()
