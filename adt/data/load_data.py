import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
from datetime import datetime
import glob

def ensure_directory_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("Created directory:", dir_path)
    else:
        print("Directory already exists:", dir_path)

def create_sliding_windows(data, window_size):
    n = data.shape[0]
    return data[np.arange(window_size)[None, :] + np.arange(n - window_size)[:, None]]

def create_window_labels(label_array, window_size):
    n = len(label_array)
    windows = label_array[np.arange(window_size)[None, :] + np.arange(n - window_size)[:, None]]
    return [1 if np.sum(w) > 0 else 0 for w in windows]

def flatten_windows(windows):
    return windows.reshape(windows.shape[0], windows.shape[1] * windows.shape[2])

def save_processed_data(output_dir, x_normal_scaled, x_attack_scaled, windows_normal_flatten, windows_attack_flatten,y_labels):
    np.save(os.path.join(output_dir, "x_attack_scaled.npy"), x_attack_scaled)
    np.save(os.path.join(output_dir, "x_normal_scaled.npy"), x_normal_scaled)
    np.save(os.path.join(output_dir, "windows_normal_flatten.npy"), windows_normal_flatten)
    np.save(os.path.join(output_dir, "windows_attack_flatten.npy"), windows_attack_flatten)
    np.save(os.path.join(output_dir, "windows_attack_labels.npy"), y_labels)
    print("Data preparation complete. Files saved in:", output_dir)


def prepare_swat(window_size=12):
    output_dir = os.path.join(os.getcwd(), "processed_data","SWaT")
    ensure_directory_exists(output_dir)
    f1 = pd.read_excel("./dataset/SWaT/SWaT_Dataset_Normal_v1.xlsx")
    normal = pd.DataFrame(f1).drop(f1.columns[[0, -1]], axis=1).astype(float)
    f2 = pd.read_excel("./dataset/SWaT/SWaT_Dataset_Attack_v0.xlsx")
    attack = pd.DataFrame(f2)
    labels = attack[attack.columns[-1]].values.tolist()
    y_all = np.array([0 if i == "Normal" else 1 for i in labels])
    attack = attack.drop(attack.columns[[0, -1]], axis=1).astype(float)
    all_data = pd.DataFrame(np.vstack((normal, attack)))
    min_max_scaler = preprocessing.MinMaxScaler()
    all_data_scaled = min_max_scaler.fit_transform(all_data.values)
    x_normal_scaled = pd.DataFrame(all_data_scaled[:495000, :])
    x_attack_scaled = pd.DataFrame(all_data_scaled[495000:, :])
    windows_normal = create_sliding_windows(x_normal_scaled.values, window_size)
    windows_attack = create_sliding_windows(x_attack_scaled.values, window_size)
    y_windows = y_all[np.arange(window_size)[None, :] + np.arange(len(y_all) - window_size)[:, None]]
    y_labels = [1 if np.sum(i) > 0 else 0 for i in y_windows]
    print("SWaT normal windows shape:", windows_normal.shape, "attack windows shape:", windows_attack.shape, "label",
          len(y_labels), "anomalies:", np.sum(y_labels))
    windows_normal_flatten = flatten_windows(windows_normal)
    windows_attack_flatten = flatten_windows(windows_attack)
    print("Flattened normal windows shape:", windows_normal_flatten.shape, "flattened attack windows shape:", windows_attack_flatten.shape)
    save_processed_data(output_dir, x_normal_scaled, x_attack_scaled, windows_normal_flatten, windows_attack_flatten,
                        y_labels)

def prepare_wadi(window_size=12):
    output_dir = os.path.join(os.getcwd(), "processed_data", "WADI")
    ensure_directory_exists(output_dir)
    f1 = pd.read_csv(
        "./dataset/WADI/WADI.A1_9_Oct_2017/WADI_14days.csv",skiprows=[0,1,2,3], sep=',',skip_blank_lines=True )
    normal = f1.drop(f1.columns[[0,1,2,50,51,86,87]],axis=1) # drop the empty columns and the date/time columns
    normal = normal.astype(float)
    normal = pd.DataFrame(normal).fillna(0)
    # normalize
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_normal_scaled = min_max_scaler.fit_transform(normal.values)
    #print(normal.isnull().sum().sum())
    f2 = pd.read_csv("./dataset/WADI/WADI.A1_9_Oct_2017/WADI_attackdata.csv",sep=",")
    attack = pd.DataFrame(f2)
    labels = []
    attack.reset_index()
    for index, row in attack.iterrows():
        date_temp = row['Date']
        date_mask = "%m/%d/%Y"
        date_obj = datetime.strptime(date_temp, date_mask)
        time_temp = row['Time']
        time_mask = "%I:%M:%S.%f %p"
        time_obj = datetime.strptime(time_temp, time_mask)

        if date_obj == datetime.strptime('10/9/2017', '%m/%d/%Y'):
            if time_obj >= datetime.strptime('7:25:00.000 PM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                    '7:50:16.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue

        if date_obj == datetime.strptime('10/10/2017', '%m/%d/%Y'):
            if time_obj >= datetime.strptime('10:24:10.000 AM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                    '10:34:00.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj >= datetime.strptime('10:55:00.000 AM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                    '11:24:00.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj >= datetime.strptime('11:30:40.000 AM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                    '11:44:50.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj >= datetime.strptime('1:39:30.000 PM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                    '1:50:40.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj >= datetime.strptime('2:48:17.000 PM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                    '2:59:55.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj >= datetime.strptime('5:40:00.000 PM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                    '5:49:40.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj >= datetime.strptime('10:55:00.000 AM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                    '10:56:27.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue

        if date_obj == datetime.strptime('10/11/2017', '%m/%d/%Y'):
            if time_obj >= datetime.strptime('11:17:54.000 AM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                    '11:31:20.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj >= datetime.strptime('11:36:31.000 AM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                    '11:47:00.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj >= datetime.strptime('11:59:00.000 AM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                    '12:05:00.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj >= datetime.strptime('12:07:30.000 PM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                    '12:10:52.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj >= datetime.strptime('12:16:00.000 PM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                    '12:25:36.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj >= datetime.strptime('3:26:30.000 PM', '%I:%M:%S.%f %p') and time_obj <= datetime.strptime(
                    '3:37:00.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue

        labels.append('Normal')
    labels=np.array([1 if i=="Attack" else 0 for i in labels])
    # Drop the empty and date/time columns
    attack = attack.drop(attack.columns[[0, 1, 2, 50, 51, 86, 87]], axis=1)
    #normalized all data together and then divide them
    all_data=pd.concat([normal, attack])
    min_max_scaler = preprocessing.MinMaxScaler()
    all_data_scaled = min_max_scaler.fit_transform(all_data.values)
    n_normal = normal.shape[0]
    x_normal_scaled=all_data_scaled[:n_normal,:]
    x_attack_scaled=all_data_scaled[n_normal:,:]
    windows_normal = create_sliding_windows(x_normal_scaled, window_size)
    windows_attack = create_sliding_windows(x_attack_scaled, window_size)
    y_windows = labels[np.arange(window_size)[None, :] + np.arange(len(labels) - window_size)[:, None]]
    y_labels = [1 if np.sum(i) > 0 else 0 for i in y_windows]
    print("WADI normal windows shape:", windows_normal.shape, "attack windows shape:", windows_attack.shape, "label",
          len(y_labels), "anomalies:", np.sum(y_labels))
    # # flatten
    windows_normal_flatten = flatten_windows(windows_normal)
    windows_attack_flatten = flatten_windows(windows_attack)
    print("Flattened normal windows shape:", windows_normal_flatten.shape, "flattened attack windows shape:", windows_attack_flatten.shape)
    save_processed_data(output_dir, x_normal_scaled, x_attack_scaled, windows_normal_flatten, windows_attack_flatten,
                        y_labels)

def prepare_hai(window_size=12):
    output_dir = os.path.join(os.getcwd(), "processed_data", "HAI")
    ensure_directory_exists(output_dir)
    csv_files = glob.glob(
        os.path.join("./dataset/HAI//hai_21.03", "*.csv"))
    normal_data=pd.read_csv(csv_files[5])
    for f in csv_files[6:]:
        normal_data=pd.concat([normal_data, pd.read_csv(f)])
    x_normal = normal_data.drop(normal_data.columns[[0, -1, -2, -3, -4]], axis=1)
    attack_data=pd.read_csv(csv_files[0])
    for f in csv_files[1:5]:
        attack_data=pd.concat([attack_data, pd.read_csv(f)])
    y_labels=np.array(attack_data[attack_data.columns[-4]].values.tolist())
    x_attack=attack_data.drop(attack_data.columns[[0, -1, -2, -3, -4]], axis=1)
    all_data=pd.concat([x_normal, x_attack])
    min_max_scaler = preprocessing.MinMaxScaler()
    all_data_scaled = min_max_scaler.fit_transform(all_data.values)
    n_normal = x_normal.shape[0]
    x_normal_scaled=all_data_scaled[:n_normal,:]
    x_attack_scaled=all_data_scaled[n_normal:,:]
    #exclude constant features in training
    constant_columns=[]
    for i,v in enumerate(x_normal_scaled.T):
        if len(np.unique(v))==1:
            constant_columns.append(i)
    x_normal_scaled = np.delete(x_normal_scaled, constant_columns, axis=1)
    x_attack_scaled=np.delete(x_attack_scaled, constant_columns, axis=1)
    windows_normal = create_sliding_windows(x_normal_scaled, window_size)
    windows_attack = create_sliding_windows(x_attack_scaled, window_size)
    y_windows = y_labels[np.arange(window_size)[None, :] + np.arange(len(y_labels) - window_size)[:, None]]
    y_labels = [1 if np.sum(i) > 0 else 0 for i in y_windows]
    print("HAI normal windows shape:", windows_normal.shape, "attack windows shape:", windows_attack.shape, "label",
          len(y_labels), "anomalies:", np.sum(y_labels))
    windows_normal_flatten = flatten_windows(windows_normal)
    windows_attack_flatten = flatten_windows(windows_attack)
    print("Flattened normal windows shape:", windows_normal_flatten.shape, "flattened attack windows shape:", windows_attack_flatten.shape)
    save_processed_data(output_dir, x_normal_scaled, x_attack_scaled, windows_normal_flatten, windows_attack_flatten,
                        y_labels)

def prepare_yahoo(window_size=12, stride=1):
    def min_max_normalization(target_data, centred_data):
        min_v = np.min(centred_data)
        max_v = np.max(centred_data)
        result = (target_data - min_v) / (max_v - min_v)
        return result

    def sliding_window_1d(target, labels, window_size=10, stripe=1):
        x = []
        y = []
        for i in range(0, len(target) - window_size + 1, stripe):
            x.append(target[i:i + window_size])
            y.append(labels[i:i + window_size])
        return np.array(x), np.array(y)
    output_dir = os.path.join(os.getcwd(), "processed_data", "Yahoo")
    ensure_directory_exists(output_dir)
    csv_files = glob.glob(
        os.path.join("./dataset/Yahoo/A1Benchmark", "*.csv")
    )
    csv_files = sorted(csv_files, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
    x_all, y_all = [], []
    x_normal, y_normal = [], []
    for f in csv_files:
        df = pd.read_csv(f)
        values = df["value"].tolist()
        labels = df["is_anomaly"].tolist()
        x_all.extend(values)
        y_all.extend(labels)
        for i, lab in enumerate(labels):
            if lab == 0:
                x_normal.append(values[i])
                y_normal.append(0)
    x_all = np.array(x_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int32)
    x_normal = np.array(x_normal, dtype=np.float32)
    y_normal = np.array(y_normal, dtype=np.int32)
    normalized_x_normal = min_max_normalization(x_normal, x_all)
    normalized_x_all = min_max_normalization(x_all, x_all)
    x_seq_normal, y_seq_normal = sliding_window_1d(normalized_x_normal, y_normal, window_size, stride)
    x_seq_all, y_seq_all = sliding_window_1d(normalized_x_all, y_all, window_size, stride)
    windows_normal_flatten = x_seq_normal.reshape(x_seq_normal.shape[0], window_size)
    windows_attack_flatten = x_seq_all.reshape(x_seq_all.shape[0], window_size)
    y_labels = []
    for w in y_seq_all:
        if np.sum(w) > 0:
            y_labels.append(1)
        else:
            y_labels.append(0)
    y_labels = np.array(y_labels, dtype=np.int32)
    x_normal_2d = normalized_x_normal.reshape(-1, 1)
    x_all_2d = normalized_x_all.reshape(-1, 1)

    print("Yahoo dataset shape (normal windows):", windows_normal_flatten.shape,
          "all windows:", windows_attack_flatten.shape,
          "labels:", y_labels.shape, "anomalies in windows:", np.sum(y_labels))
    save_processed_data(
        output_dir,
        x_normal_2d,
        x_all_2d,
        windows_normal_flatten,
        windows_attack_flatten,
        y_labels
    )
