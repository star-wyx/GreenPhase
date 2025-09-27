import numpy as np
from matplotlib import pyplot as plt
from skimage.util import view_as_windows
import os
import gc
from scipy.signal import find_peaks
from FeatureTest import FeatureTest
from numba import njit, prange


def Shrink(X, shrinkArg):
    filter_width = shrinkArg['filter_width']
    stride = shrinkArg['stride']
    padding = shrinkArg['padding']
    X = np.pad(X, ((0, 0), (0, 0), (0, padding), (0, 0)), mode='constant')
    X = view_as_windows(X, (1, 3, filter_width, 1), (1, 3, stride, 1))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)


def Concat(X, concatArg):
    return X


def plot_engery(pixelHop, title, path):
    cumulative_energies = np.cumsum(pixelHop.Energy['Layer0'][0])
    x_values = np.arange(1, len(cumulative_energies) + 1)
    plt.plot(x_values, cumulative_energies, marker='o')
    plt.title(title)
    if path is not None:
        plt.savefig(path)
    # plt.show()
    plt.close()


def plot_train_val_rank(dft, val_dft, path=None):
    training_rank = [dft.dim_rank[dim] for dim in range(dft.dim)]
    validation_rank = [val_dft.dim_rank[dim] for dim in range(val_dft.dim)]
    plt.figure(figsize=(8, 6))
    plt.scatter(training_rank, validation_rank, color='blue', alpha=1)

    plt.title('Feature Rank in Training vs. Validation')
    plt.xlabel('Training Rank')
    plt.ylabel('Validation Rank')
    if path is not None:
        plt.savefig(path)
    # plt.show()
    plt.close()


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def pseudo_label_sample(input_shape,
                        wave_locations,
                        win_len,
                        step=1,
                        delta_t=10,
                        num_sample=50000,
                        high_percent=0.33,
                        low_percent=0.33,
                        save_path=None):
    """
    Generate pseudo-labels for sliding windows around p-wave / s-wave arrivals,
    then subsample up to n_samples_per_label for *each unique* label value.

    Args:
        input_shape (tuple): (N_orig, C, L_orig)
        wave_locations (np.ndarray): shape (N_orig,), p-wave / s-wave sample indices
        step (int): sliding window step
        num_sample (int): The total number of samples
        save_path (str, optional): directory to save histogram

    Returns:
        sub_labels (np.ndarray): 1D float32 array
        sub_coords (np.ndarray): 2D int32 array of [sample_idx, center_idx]
    """
    N_orig, _, seq_len = input_shape
    center_off = win_len // 2

    # --- 1. build window starts & centers (int32) ---
    n_windows = (seq_len - win_len) // step + 1
    starts = np.arange(n_windows, dtype=np.int32) * step  # (n_windows,)
    centers = starts + center_off  # (n_windows,)

    # broadcast into full grids
    all_starts = np.broadcast_to(starts, (N_orig, n_windows))
    all_centers = np.broadcast_to(centers, (N_orig, n_windows))
    sample_idxs = np.repeat(np.arange(N_orig, dtype=np.int32),
                            n_windows).reshape(N_orig, n_windows)

    # --- 2. compute labels (float32) ---
    wave = wave_locations.astype(np.int32)[:, None]  # (N_orig,1)
    L = wave - all_starts  # (N_orig,n_windows)
    R = (all_starts + win_len) - wave

    within = (wave >= all_starts) & (wave <= all_starts + win_len)

    labels = np.zeros_like(L, dtype=np.float32)
    # perfect center
    centered = np.abs(L - R) <= delta_t
    labels[within & centered] = 1.0

    # partial
    mask = within & ~centered
    min_lr = np.minimum(L[mask], R[mask]).astype(np.float32)
    max_lr = np.maximum(L[mask], R[mask]).astype(np.float32)
    labels[mask] = min_lr / (max_lr + np.finfo(np.float32).eps)

    # wave_locations==-1（noise event), label=0
    noise_mask = (wave_locations == -1)
    if noise_mask.any():
        labels[noise_mask, :] = 0.0

    # free intermediates
    del L, R, within, centered, min_lr, max_lr
    gc.collect()

    # --- 3. flatten arrays ---
    labels_flat = labels.ravel()
    coords_flat = np.stack([
        sample_idxs.ravel(),
        all_centers.ravel()
    ], axis=1).astype(np.int32)

    # free grids
    del labels, sample_idxs, all_centers, all_starts
    gc.collect()

    # --- 4. subsample per unique label ---
    high_n = int(num_sample * high_percent)  # label >= 0.8
    low_n = int(num_sample * low_percent)  # label == 0
    mid_n = num_sample - high_n - low_n  # rest

    high_idxs = np.nonzero(labels_flat >= 0.8)[0]
    low_idxs = np.nonzero(labels_flat == 0)[0]
    mid_idxs = np.nonzero((labels_flat > 0) & (labels_flat < 0.8))[0]

    selected = []
    if high_idxs.size > high_n:
        high_idxs = np.random.choice(high_idxs, high_n, replace=False)
    else:
        low_n = high_idxs.size
        mid_n = high_idxs.size
    selected.append(high_idxs)

    if low_idxs.size > low_n:
        low_idxs = np.random.choice(low_idxs, low_n, replace=False)
    selected.append(low_idxs)

    if mid_idxs.size > mid_n:
        mid_idxs = np.random.choice(mid_idxs, mid_n, replace=False)
    selected.append(mid_idxs)

    selected = np.concatenate(selected)
    perm = np.random.permutation(selected.shape[0])
    selected = selected[perm]

    # --- 5. gather and shuffle ---
    sub_labels = labels_flat[selected]
    sub_coords = coords_flat[selected]

    perm = np.random.permutation(sub_labels.shape[0])
    sub_labels = sub_labels[perm]
    sub_coords = sub_coords[perm]

    # optional histogram
    if save_path:
        plt.figure()
        plt.hist(sub_labels, bins=100, edgecolor='black')
        plt.title('Pseudo-label Distribution')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    return sub_labels, sub_coords


def extract_saab_windows(feat, coords, win_len, stride=1):
    """
    feat: (N, 1, L, C)
    coords: (M, 2) array of [sample_idx, center_pos]
    win_size: feature window size of each layer
    Returns:
        (M, win_size[set_idx], C) array of windows
    """
    start_indices = coords[:, 1] - win_len // 2
    offsets = np.arange(0, win_len, stride)

    final_col_indices = start_indices[:, np.newaxis] + offsets
    sample_indices = coords[:, 0]
    result = feat[sample_indices[:, np.newaxis], :, final_col_indices, :]
    return result


@njit(parallel=True, fastmath=True)
def _energy_numba(prefix, coords, half, L):
    M = coords.shape[0]
    C = prefix.shape[1]
    out = np.empty((M, 1), np.float32)
    for i in prange(M):
        sidx = int(coords[i, 0])
        center = int(coords[i, 1])
        start = center - half
        end = center + half
        left = 0.0
        right = 0.0
        for c in range(C):
            left += prefix[sidx, c, center] - prefix[sidx, c, start]
            right += prefix[sidx, c, end + 1] - prefix[sidx, c, center + 1]
        out[i, 0] = (right - left) / (half * C)
    return out


def compute_window_energy(data, coords, win_len=513):
    half = win_len // 2
    N, C, L = data.shape
    sq = data.astype(np.float32) ** 2
    prefix = np.concatenate(
        (np.zeros((N, C, 1), dtype=np.float32), np.cumsum(sq, axis=-1)),
        axis=-1
    )
    sidx = coords[:, 0].astype(np.int64)
    center = coords[:, 1].astype(np.int64)
    start = center - half
    end = center + half
    if np.any(start < 0) or np.any(end >= L):
        raise IndexError("Some windows are out of range")
    return _energy_numba(prefix, coords.astype(np.int64), half, L)


def plot_xgb_learning_curve(eval_result, eval_metric='logloss', path=None):
    plt.figure()
    plt.plot(eval_result['train'][eval_metric], label='train')
    plt.plot(eval_result['val'][eval_metric], label='val')
    plt.xlabel('Iteration')
    plt.ylabel(eval_metric)
    plt.legend()
    if path is not None:
        plt.savefig(path)
    plt.close()


def average_pooling(x, num_transform):
    if num_transform == 0:
        return x

    block_size = 2 ** num_transform
    scale_factor = np.power(np.sqrt(2), -num_transform)

    N, C, L = x.shape
    new_L = L // block_size

    avg = x.reshape(N, C, new_L, block_size).sum(axis=-1)
    return (avg * scale_factor).astype(np.float32)


def map_label_after_pooling(y, num_transform):
    return y // (2 ** num_transform)


def post_process_prob(prob_trace, peak_dist=300, thr_mul=0.7):
    peaks, _ = find_peaks(prob_trace, distance=peak_dist)
    if peaks.size == 0:
        return None
    max_val = prob_trace[peaks].max()
    valid = peaks[prob_trace[peaks] >= thr_mul * max_val]
    return None if valid.size == 0 else int(valid[0])


def build_center_indices(N, L, win_len, step, roi_bounds=None):
    """
    Generate (sample_idx, center_idx) index table.
    roi_bounds:  None            → full sequence traversal
                 (N, 2) ndarray → [left, right] (closed interval) for each sample
    """
    j_min, j_max = win_len // 2, L - win_len // 2  # valid center range
    if roi_bounds is None:
        center_grid = np.arange(j_min, j_max, step, dtype=np.int32)
        I, J = np.indices((N, center_grid.size))
        J = J + center_grid[0]  # center_grid[0] = j_min
        return np.stack((I, J), axis=-1).reshape(-1, 2)

    # scan ROI only
    indices_list = []
    for n in range(N):
        left, right = roi_bounds[n]
        left = max(left, j_min)
        right = min(right, j_max - 1)
        if left > right:
            continue
        centers = np.arange(left, right + 1, step, dtype=np.int32)
        idx = np.column_stack((np.full_like(centers, n), centers))
        indices_list.append(idx)
    if len(indices_list) == 0:
        return None
    return np.concatenate(indices_list, axis=0)


def plot_rft_lnt_features(X_train, y_train, rft, lnt, idx, save_path):
    plt_rft = FeatureTest('rmse')
    plt_rft.fit(X_train, y_train, n_bins=16, outliers=True)
    feat_idx = rft.n_selected
    label_dims = {f'Features set {idx}': list(range(feat_idx))}
    label_dims['LNT Features'] = list(range(feat_idx, feat_idx + lnt.dim))

    plt_rft.plot(label_dims=label_dims,
                 path=os.path.join(save_path, f"level_{idx}_train_rft_lnt.png"))


def batch_transform_saab_on_demand(X_source, indices_to_process, model_instance, level, win_len, saab_kernel,
                                   saab_stride, batch_size=50000):
    """
    Extracts SAAB feature windows for specific points of interest using a memory-efficient batching strategy.

    Args:
        X_source (np.ndarray): The source data array (e.g., X_haar) with shape (N, C, L).
        indices_to_process (np.ndarray): A (M, 2) array of [wave_idx, center_pos] coordinates.
        model_instance (Model): The instance of the Model class.
        level (int): The current processing level.
        win_len (int): The desired final feature window length (for RFT/LNT).
        saab_kernel (int): The width of the SAAB kernel.
        saab_stride (int): The stride for the final feature sampling.
        batch_size (int): The number of samples to process in each batch to control memory usage.

    Returns:
        np.ndarray: An array containing the extracted feature windows.
    """
    num_samples = indices_to_process.shape[0]
    if num_samples == 0:
        return np.array([])

    # Prepare the destination array once at the beginning
    pixelhop_model = model_instance.pixelhop_lst[level]
    num_saab_features = pixelhop_model.par['Layer0'][0].Kernels.shape[0]
    rft_win_len_strided = len(np.arange(0, win_len, saab_stride))
    saab_features = np.zeros((num_samples, rft_win_len_strided, num_saab_features), dtype=np.float32)

    # Loop over the indices in smaller batches
    for i in range(0, num_samples, batch_size):
        batch_indices = indices_to_process[i:i + batch_size]

        # --- Vectorized segment extraction to avoid python loops ---
        buffer = saab_kernel - 1
        required_len = win_len + 2 * buffer
        half_len = required_len // 2

        batch_size_actual = len(batch_indices)
        num_channels = X_source.shape[1]

        # 1. Create destination array and calculate all source/destination indices
        batched_raw_segments = np.zeros((batch_size_actual, num_channels, required_len), dtype=X_source.dtype)
        wave_indices = batch_indices[:, 0]
        center_positions = batch_indices[:, 1]

        starts = center_positions - half_len
        offsets = np.arange(required_len)
        src_cols = starts[:, np.newaxis] + offsets

        # 2. Create a mask for valid indices that are within the bounds of X_source
        valid_mask = (src_cols >= 0) & (src_cols < X_source.shape[-1])

        # 3. Get the coordinates for the valid source and destination elements
        dest_row_indices, dest_col_indices = np.where(valid_mask)

        src_wave_indices = wave_indices[dest_row_indices]
        src_col_indices = src_cols[dest_row_indices, dest_col_indices]

        # 4. Perform the copy in a single vectorized operation
        # Note: The result of advanced indexing on (N,C,L) with (idx_N, idx_L) is (C, num_valid), so we transpose.
        if dest_row_indices.size > 0:
            batched_raw_segments[dest_row_indices, :, dest_col_indices] = X_source[src_wave_indices, :, src_col_indices]
        # --- End of vectorized extraction ---

        # Transform the batch
        batched_feature_windows = model_instance.transform_saab(batched_raw_segments, idx=level)

        # Process the resulting feature windows
        fw_len = batched_feature_windows.shape[2]
        center_of_fw = fw_len // 2
        half_win = win_len // 2
        start_fw = center_of_fw - half_win
        end_fw = start_fw + win_len

        if start_fw < 0 or end_fw > fw_len:
            raise ValueError(
                f"Cannot extract central window. Generated feature length ({fw_len}) is smaller than required ({win_len}).")

        centered_fw_batch = batched_feature_windows[:, :, start_fw:end_fw, :]
        strided_batch = centered_fw_batch[:, 0, ::saab_stride, :]

        # Assign the batch results to the final array
        saab_features[i:i + batch_size] = strided_batch

        # Explicitly delete large intermediate arrays to free memory
        del batch_indices, batched_raw_segments, batched_feature_windows, centered_fw_batch, strided_batch
        gc.collect()

    return saab_features
