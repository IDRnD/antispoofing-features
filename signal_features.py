import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path, PosixPath

import numpy as np
from tqdm import tqdm
from hurst import compute_Hc
from skimage.util import view_as_windows
from librosa.feature import zero_crossing_rate, rms

from config import Config
from tools.feature_extractor import SignalLoader, stat_features


def get_repeats(signal: np.ndarray,
                num_features: int = 16,
                threshold: int = 1,
                dtype=np.uint8):
    """
    Counts the number of plateaus of different lengths in the raw signal

    Parameters:
    signal: np.ndarray - input 1D signal
    num_features: int - maximim length of detected repeats
    threshold: int - minimum length of detected repeats
    dtype: type of extracted features

    Returns:
    repeat_counter: np.ndarray - plateau features
    feature_names: list - names of extracted features
    """

    repeat_counter = np.zeros(shape=(num_features,), dtype=dtype)
    file_conv = np.convolve(signal, [-1, 1])
    file_conv = np.where(file_conv == 0)[0]
    file_conv = file_conv - np.arange(start=0, stop=len(file_conv))
    counts = np.unique(file_conv, return_counts=True)[1] - threshold
    repeats_counter = Counter(counts[(counts > 0) & (counts < num_features + threshold)])

    for key, val in repeats_counter.items():
        repeat_counter[key - 1] = val

    feature_names = [f'repeats_{i}' for i in range(num_features)]
    return repeat_counter, feature_names


def get_signal_stats(signal: np.ndarray, signal_features_config: dict):
    """
    Extracts various statistics from the raw signal

    Parameters:
    signal: np.ndarray - input 1D signal
    signal_features_config: dict - stat. features that should be extracted

    Returns:
    features: np.ndarray - extracted stat. features
    feature_names: list - names of extracted features
    """

    features, feature_names = [], []
    file_types = {
        'signal': signal,
        'abs': np.abs(signal),
        'diff': np.diff(signal),
        'zero_cross': zero_crossing_rate(signal)[0],
        'rms': rms(signal)[0]
    }

    for signal_feature_name, config in signal_features_config.items():
        for stat_feature_key, included in config.items():
            if not included:
                continue
            
            stat_feature_name = stat_feature_key.split('_')[0]
            feature = stat_features[stat_feature_name](file_types[signal_feature_name])
            if stat_feature_key == 'mode_val':
                feature = feature.mode[0]
            elif stat_feature_key == 'mode_cnt':
                feature = feature.count[0]

            features.append(feature)

            name = f'{signal_feature_name}_{stat_feature_key}'
            feature_names.append(name)

    return features, feature_names


def get_symmetry_diff(signal: np.ndarray, eps: float = 1e-8):
    """
    Calculates samples symmetry relative to zero

    Parameters:
    signal: np.ndarray - input 1D signal
    eps: float - epsilon constant

    Returns:
    features: np.ndarray - extracted stat. features
    feature_names: list - names of extracted features
    """

    pos_part = signal[signal >= 0]
    neg_part = signal[signal < 0]
    symmetry_diff = np.abs(pos_part.sum() + neg_part.sum())

    min_len = min(len(pos_part), len(neg_part))
    pos_part = pos_part[: min_len]
    neg_part = neg_part[: min_len]
    diff = pos_part + neg_part

    num_equal_bins = len(np.where(diff == 0)[0]) / (len(diff) + eps)
    num_diff_bins = 1 - num_equal_bins
    symmetry_diff = symmetry_diff / (np.abs(signal).max() + eps)

    features = [num_equal_bins, num_diff_bins, symmetry_diff]
    feature_names = ['symm_num_equal_bins', 'symm_num_diff_bins', 'symm_abs_diff']

    return features, feature_names


def get_hurst_exp(signal: np.ndarray):
    """
    Extracts Hurst coefficients from the raw signal

    Parameters:
    signal: np.ndarray - input 1D signal

    Returns:
    features: np.ndarray - extracted stat. features
    feature_names: list - names of extracted features
    """

    H, c, _ = compute_Hc(signal, kind='random_walk', simplified=True)
    features = np.array([H, c])
    feature_names = ['hurst_H', 'hurst_c']

    return features, feature_names


def calc_window_stats(chunks, feature, name, num_bins):
    if name == 'mode_val':
        chunks = feature(chunks, axis=1).mode[0]
    elif name == 'mode_cnt':
        chunks = feature(chunks, axis=1).count[0]
    else:
        chunks = feature(chunks, axis=1)

    diff = np.abs(np.diff(chunks))
    hist = np.histogram(diff, bins=num_bins)

    return hist


def get_window_stats(signal: np.ndarray,
                     win_len: int = 4096,
                     step: int = 256,
                     num_bins: int = 5):
    """
    Slices the raw signal into a set of windows and calculates statistical parameters for each window

    Parameters:
    signal: np.ndarray - input 1D signal
    win_len: int - length of the window
    step: int - step size
    num_bins: int - number of bins in the generated histogram

    Returns:
    features: np.ndarray - extracted stat. features
    feature_names: list - names of extracted features
    """

    wind_stats_config = {
        'min': True,
        'max': True,
        'mean': True,
        'std': True,
        'skew': True,
        'kurtosis': True,
        'mode_val': True,
        'mode_count': True,
        'iqr': True,
        'sem': True,
        'mad': True
    }

    features = []
    feature_names = []

    chunks = view_as_windows(signal, win_len, step)
    for name, included in wind_stats_config.items():
        if not included:
            continue

        feature_name = name.split('_')[0]
        feature = stat_features[feature_name]

        bin_count, bin_val = calc_window_stats(chunks, feature, name, num_bins)
        features.extend([bin_count, bin_val])

        names = []
        for i in range(1, len(bin_count) + 1):
            names.append(f'wind_stats_{name}_cnt_{i}')

        for i in range(1, len(bin_val) + 1):
            names.append(f'wind_stats_{name}_val_{i}')

        feature_names.extend(names)

    return np.hstack(features), feature_names


def calc_statistics(paths: list,
                    feature_name: str,
                    feature_type: str,
                    stat_func,
                    save_path: PosixPath = None,
                    save_feature_names: bool = False,
                    dtype=np.float32):
    """
    Calculates stat. features on the raw audio signal

    Parameters:
    paths: np.ndarray - paths to the audio files
    feature_name: str - name of the processed feature
    feature_type: str - type of the processed feature (train, dev or val)
    stat_func: function for extraction of stat. feature
    save_path: PosixPath - path for saving of calculated features
    save_feature_names: bool - save names of calculated features or not
    dtype: type of extracted features

    Returns:
    statistics: np.ndarray - extracted stat. features
    """

    assert feature_type in ('train', 'dev', 'val')

    statistics = []
    with Pool(processes=Config.num_proc) as p:
        with tqdm(total=len(paths)) as pbar:
            for stat, names in p.imap(stat_func, paths):
                statistics.append(stat)
                pbar.update()

    if save_path is not None and save_feature_names:
        (save_path / 'feature_names').mkdir(exist_ok=True, parents=True)
        np.save(save_path / 'feature_names' / f'{feature_name}_names', names)

    statistics = np.array(statistics, dtype=dtype)
    if save_path is not None:
        (save_path / feature_type).mkdir(exist_ok=True, parents=True)
        np.save(save_path / f'{feature_type}/{feature_name}', statistics)

    return statistics


if __name__ == '__main__':
    root_dir = Path(__file__).parent
    file_path = root_dir / 'tests' / 'LA_T_1000137.flac'
    config_path = root_dir / 'configs'
    with open(config_path / 'signal_features.json', 'r') as config:
        signal_features_config = json.load(config)

    features_config = {
        'repeats': {'stat_func': SignalLoader(get_repeats, normalize=True, **{'num_features': 16}), 'dtype': np.uint8},
        'stats': {'stat_func': SignalLoader(get_signal_stats, normalize=True, **{'signal_features_config': signal_features_config})},
        'symmetry': {'stat_func': SignalLoader(get_symmetry_diff)},
        'wind_stats': {'stat_func': SignalLoader(get_window_stats, normalize=True)},
        'hurst': {'stat_func': SignalLoader(get_hurst_exp)}
    }

    paths = [file_path]
    feature_type = 'train'
    for feature_name, config in features_config.items():
        statistics = calc_statistics(paths=paths,
                                     feature_name=feature_name,
                                     feature_type=feature_type,
                                     **config)
                                     
        statistics_test = np.load(root_dir / 'tests' / 'signal_features' / f'LA_T_1000137_{feature_name}.npy')
        assert np.all(statistics == statistics_test), f'Test for {feature_name} not passed'
        print(feature_name, statistics.shape)
        
    print('OK')
