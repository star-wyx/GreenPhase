import h5py
import numpy as np
from tqdm import tqdm
import os
from scipy.signal import butter, sosfiltfilt
from joblib import Parallel, delayed
import pandas as pd
import gc


def stead_to_npy_stream(path):
    h5_path = os.path.join(path, 'merge.hdf5')
    csv_path = os.path.join(path, 'merge.csv')

    df = (pd.read_csv(csv_path)
          .fillna({'p_arrival_sample': -1, 's_arrival_sample': -1})
          .astype({'p_arrival_sample': 'int32', 's_arrival_sample': 'int32'}))
    names = df['trace_name'].tolist()
    N = len(names)

    np.save(os.path.join(path, 'P.npy'), df['p_arrival_sample'].values)
    np.save(os.path.join(path, 'S.npy'), df['s_arrival_sample'].values)
    np.save(os.path.join(path, 'Event_ID.npy'), np.array(names, dtype='U32'))

    print(f"Reading {N} traces and streaming to X.npy ...")
    X_mm = np.memmap(os.path.join(path, 'X.npy'), mode='w+', dtype='float32', shape=(N, 6000, 3))
    with h5py.File(h5_path, 'r') as h5_in:
        for i, name in enumerate(tqdm(names, desc="stream copy")):
            X_mm[i] = h5_in[f'data/{name}'][:]

    del X_mm
    print("Done! Files written: X.npy, P.npy, S.npy, Event_ID.npy")


def generate_train_test(path, trace_name, fs=100, fmin=1., fmax=45., n_jobs=-1):
    """
    ------------------------------------------------------------------
    path      : data directory (contains merge.hdf5 / merge.csv)
    test_trace  : test set trace file
    generate_train  : True = generate train set, False = generate test set
    fs        : sampling rate
    fmin,fmax : bandpass filter limits (Hz)
    n_jobs    : number of cores, -1 = use all cores
    ------------------------------------------------------------------
    """

    # ---------- 1. load array ----------
    test_trace = np.load(os.path.join(path, trace_name)).astype(str)
    df = (pd.read_csv(os.path.join(path, 'merge.csv'))
          .fillna({'p_arrival_sample': -1, 's_arrival_sample': -1})
          .astype({'p_arrival_sample': 'int32', 's_arrival_sample': 'int32'}))
    num_traces = len(df)

    X = np.memmap(os.path.join(path, 'X.npy'), mode='r', dtype='float32', shape=(num_traces, 6000, 3))
    P = np.load(os.path.join(path, 'P.npy'))
    S = np.load(os.path.join(path, 'S.npy'))
    EVID = np.load(os.path.join(path, 'Event_ID.npy'))

    E, N, Z = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    print(f'Total records to process: {X.shape[0]}')
    print(f'Waveform matrix shape: {E.shape}')
    del X
    gc.collect()

    # ---------- 2. parallel bandpass filter ----------
    sos = butter(2, [fmin, fmax], btype='bandpass', fs=fs, output='sos')
    filt = lambda x: sosfiltfilt(sos, x).astype(np.float32)

    filtered_E = Parallel(n_jobs=n_jobs)(delayed(filt)(row) for row in tqdm(E, desc='filter E'))
    filtered_N = Parallel(n_jobs=n_jobs)(delayed(filt)(row) for row in tqdm(N, desc='filter N'))
    filtered_Z = Parallel(n_jobs=n_jobs)(delayed(filt)(row) for row in tqdm(Z, desc='filter Z'))
    del E, N, Z
    gc.collect()

    data = np.abs(np.stack([filtered_E, filtered_N, filtered_Z], axis=1))  # (num_traces,3,6000)
    del filtered_E, filtered_N, filtered_Z
    gc.collect()

    # ---------- 3. vectorized normalization + remove all-zero/constant entries ----------
    min_vals = data.min(axis=2)
    max_vals = data.max(axis=2)
    good_mask = (max_vals > min_vals).all(axis=1)
    data = data[good_mask]
    P, S, EVID = P[good_mask], S[good_mask], EVID[good_mask]

    g_min = data.min(axis=(1, 2))
    g_max = data.max(axis=(1, 2))
    data = (data - g_min[:, None, None]) / (g_max - g_min)[:, None, None]
    print(f'Valid samples: {data.shape[0]}')

    # ---------- 4. save final results ----------
    # X_train_90pct_all include traces without P/S arrival
    train_mask = ~np.isin(EVID, test_trace)
    print(f"Training: {train_mask.sum()} traces")
    np.save(os.path.join(path, 'X_train.npy'), data[train_mask].astype(np.float32))
    np.save(os.path.join(path, 'P_train.npy'), P[train_mask])
    np.save(os.path.join(path, 'S_train.npy'), S[train_mask])
    np.save(os.path.join(path, 'Event_train.npy'), EVID[train_mask])

    test_mask = np.isin(EVID, test_trace)
    print(f"Testing: {test_mask.sum()} traces")
    np.save(os.path.join(path, 'X_test.npy'), data[test_mask].astype(np.float32))
    np.save(os.path.join(path, 'P_test.npy'), P[test_mask])
    np.save(os.path.join(path, 'S_test.npy'), S[test_mask])
    np.save(os.path.join(path, 'Event_test.npy'), EVID[test_mask])
    print(f'All done, files saved to: {path}')


if __name__ == "__main__":
    """
    Total traces: 1,265,657
    test_trace: 126,566
    Valid traces: 1,261,002
    Training: 1,134,906
    Testing: 126,096
    """

    path = "./data"
    trace_name = 'Eqtransformer_test_trace.npy'
    # stead_to_npy_stream(path)
    # generate_train_test(path, trace_name, n_jobs=4)
