import json
import numpy as np
from tqdm import tqdm
from librosa import stft
from multiprocessing import Pool
from pathlib import Path, PosixPath

from config import Config
from tools.feature_extractor import SignalLoader, extract_spec_features, stat_features


def bispectrum_signal(signal: np.ndarray,
                      n_fft: int = 256,
                      hop_length: int = 128,
                      eps: float = 1e-8):
    """
    Calculates bispectrum of the input 1D signal
    
    Parameters:
    n_fft: int - number of dft point
    hop_length: int - step size
    eps: float - epsilon constant

    Returns:
    magnitude: np.ndarray - magnitude of the bispectrum
    phase: np.ndarray - phase of the bispectrum
    """

    _stft = stft(signal, n_fft=n_fft, hop_length=hop_length)
    nfft = _stft.shape[0]
    freq_sum = np.arange(nfft)[:, None] + np.arange(nfft)
    cut_freq = np.min(np.nonzero(np.diagonal(freq_sum) >= nfft - 1)[0])
    arg = np.arange(cut_freq)

    num = np.mean(_stft[arg, None, :] * _stft[None, arg, :] *
                  np.conjugate(_stft[freq_sum[:cut_freq, :cut_freq], :]), axis=-1)

    denum = np.sqrt(np.mean(np.abs(_stft[arg, None, :] * _stft[None, arg, :]) ** 2, axis=-1) *
                    np.mean(np.abs(_stft[freq_sum[:cut_freq, :cut_freq], :]) ** 2, axis=-1))

    bispectrum = num / (denum + eps)
    magnitude = np.abs(bispectrum)
    phase = np.arctan2(bispectrum.imag, bispectrum.real)

    return magnitude, phase


def calc_bispec_stats(magnitude: np.ndarray,
                      phase: np.ndarray,
                      config: dict,
                      stat_features_config: dict):
    """
    Calculates stat. features of the signal bispectrum

    Parameters:
    magnitude: np.ndarray - magnitude of the bispectrum
    phase: np.ndarray - phase of the bispectrum
    config: dict - spectral features that should be extracted
    stat_features_config: dict - stat. features that should be extracted

    Returns:
    features: np.ndarray - extracted stat. features
    feature_names: np.ndarray - names of extracted features
    """

    bispec_features = {
        'magnitude': magnitude,
        'phase': phase
    }
    
    magn_spec_features, magn_spec_feature_names = \
      extract_spec_features(spectrogram=magnitude, 
                            config=config, 
                            prefix='bispec')

    features, feature_names = [], []
    for bispec_feature_name, stat_config in stat_features_config.items():
        for stat_feature_key, included in stat_config.items():
            if not included:
                continue
            
            stat_feature_name = stat_feature_key.split('_')[0]
            feature = stat_features[stat_feature_name](bispec_features[bispec_feature_name], axis=None)
            if stat_feature_key == 'mode_val':
                feature = feature.mode[0]
            elif stat_feature_key == 'mode_cnt':
                feature = feature.count[0]

            features.append(feature)

            name = f'{bispec_feature_name}_{stat_feature_key}'
            feature_names.append(name)

    return (np.hstack((features, magn_spec_features)), 
            np.hstack((feature_names, magn_spec_feature_names)))


def calc_bispectrum(paths: list,
                    feature_type: str,
                    config_path: PosixPath,
                    save_path: PosixPath = None,
                    save_feature_names: bool = False,
                    num_features: int = 66,
                    nfft: int = 256,
                    hop_length: int = 128):
    """
    Calculates bispectrum features

    Parameters:
    paths: np.ndarray - paths to the audio files
    feature_type: str - type of the processed feature (train, dev or val)
    config_path: PosixPath - path to the bispectrum config
    save_path: PosixPath - path for saving of calculated features
    save_feature_names: bool - save names of calculated features or not
    num_features: int - number of extracted features
    nfft: int - number of dft point
    hop_length: int - step size

    Returns:
    statistic_features: np.ndarray - extracted bispectrum features
    """

    assert feature_type in ('train', 'dev', 'val')

    statistic_features = np.empty(shape=(len(paths), num_features), dtype=np.float32)
    get_bispec = SignalLoader(bispectrum_signal, normalize=True, **{'n_fft': nfft, 'hop_length': hop_length})
        
    with open(config_path / 'bispectrum_features.json', 'r') as f:
        config = json.load(f) 
    with open(config_path / 'bispectrum_stats.json', 'r') as f:
        stat_features_config = json.load(f)
        
    with Pool(processes=Config.num_proc) as p:
        with tqdm(total=len(paths)) as pbar:
            for i, (magnitude, phase) in enumerate(p.imap(get_bispec, paths)):
                pbar.update()
                features, names = calc_bispec_stats(magnitude, phase, config, stat_features_config)
                statistic_features[i] = features

    if save_path is not None and save_feature_names:
        (save_path / 'feature_names').mkdir(exist_ok=True, parents=True)
        np.save(save_path / 'feature_names' / 'bispec_names', names)

    if save_path is not None:
        (save_path / feature_type).mkdir(exist_ok=True, parents=True)
        np.save(save_path / f'{feature_type}/bispec_stats', statistic_features)

    return statistic_features


if __name__ == '__main__':
    root_dir = Path(__file__).parent
    file_path = root_dir / 'tests' / 'LA_T_1000137.flac'
    config_path = root_dir / 'configs'
    paths = [file_path]
    feature_type = 'train'
    
    bispec_features = calc_bispectrum(paths=paths, 
                                      feature_type=feature_type,
                                      config_path=config_path)

    bispec_test = np.load(root_dir / 'tests' / 'bispec.npy')
    assert np.all(bispec_features == bispec_test), 'Test for bispectrum not passed'
    print(bispec_features.shape)

    print('OK')
