import json
import numpy as np
from tqdm import tqdm
from pathlib import Path, PosixPath
from multiprocessing import Pool
from librosa.feature.spectral import _spectrogram
from librosa.feature import tempogram, fourier_tempogram, melspectrogram, tonnetz
from librosa.feature import mfcc, chroma_stft, chroma_cqt, chroma_cens, poly_features 

from tools.feature_extractor import SignalLoader, extract_spec_features


def calc_fbank(y: np.ndarray,
               frame_len: int = 800,
               shift: int = 400,
               nfft: int = 512,
               nfilt: int = 42,
               sr: int = 16000,
               pre_emphasis: float = 0.97,
               normalize: bool = True,
               eps: float = 1e-8):
    """
    Calculates Filter banks from the input signal

    Parameters:
    y: np.ndarray - input signal
    frame_len: float - length of the frame
    shift: float - shift of the frames
    nfft: int - number of dft point
    nfilt: int - number of filters
    sample_rate: int - sample rate of the input signal
    pre_emphasis: float - preprocessing constant
    normalize: bool - normalize fbank or not
    eps: float - epsilon constant

    Returns:
    filter_banks: np.ndarray - filter banks
    """
    
    emph_signal = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    
    signal_len = len(emph_signal)
    num_frames = int(np.ceil(np.abs(signal_len - frame_len) / shift))
    pad_signal_len = num_frames * shift + frame_len
    z = np.zeros((pad_signal_len - signal_len))
    
    # Pad Signal to make sure that all frames have equal number 
    # of samples without truncating any samples from the original signal
    pad_signal = np.append(emph_signal, z)

    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) +\
              np.tile(np.arange(0, num_frames * shift, shift), (frame_len, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Hamming window
    frames *= np.hamming(frame_len)
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    pow_frames = (mag_frames)**2 / nfft

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((nfft + 1) * hz_points / sr)

    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
            
    filter_banks = np.dot(pow_frames, fbank.T)
    # numerical stability
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB

    if normalize:
        filter_banks -= (np.mean(filter_banks, axis=0) + eps)
    
    return filter_banks


SAMPLE_RATE = 16000
feature_functions = {
    'spec': SignalLoader(_spectrogram, get_first=True),
    'mfcc': SignalLoader(mfcc, abs_val=True, **{'sr': SAMPLE_RATE}),
    'chroma_stft': SignalLoader(chroma_stft, **{'sr': SAMPLE_RATE}),
    'chroma_cqt': SignalLoader(chroma_cqt, **{'sr': SAMPLE_RATE}),
    'chroma_cens': SignalLoader(chroma_cens, **{'sr': SAMPLE_RATE}),
    'mel_spec': SignalLoader(melspectrogram, **{'sr': SAMPLE_RATE}),
    'tonnetz': SignalLoader(tonnetz, abs_val=True, **{'sr': SAMPLE_RATE}),
    'poly': SignalLoader(poly_features, abs_val=True, **{'sr': SAMPLE_RATE, 'order': 5}),
    'tempogram': SignalLoader(tempogram, abs_val=True, **{'sr': SAMPLE_RATE}),
    'fourier_tempogram': SignalLoader(fourier_tempogram, abs_val=True, **{'sr': SAMPLE_RATE}),
    'fbank': SignalLoader(calc_fbank, abs_val=True, **{'sr': SAMPLE_RATE})
}


def calc_spec_statistics(paths: list,
                         feature_name: str,
                         feature_config: dict,
                         feature_type: str,
                         save_path: PosixPath = None,
                         save_feature_names: bool = False):
    """
    Calculates stat. features of the signal spectrum

    Parameters:
    paths: np.ndarray - paths to the audio files
    feature_name: str - name of the processed spectral feature
    feature_config: dict - configuration of processed spectral feature
    feature_type: str - type of the processed feature (train, dev or val)
    save_path: PosixPath - path for saving of calculated features
    save_feature_names: bool - save names of calculated features or not

    Returns:
    statistics: np.ndarray - stat. features extracted from the spectrum of the signal
    """

    assert feature_name in feature_functions.keys()
    assert feature_type in ('train', 'dev', 'val')

    statistics = []
    stat_func = feature_functions[feature_name]
    
    with tqdm(total=len(paths)) as pbar:
        for spec in map(stat_func, paths):
            features, names = extract_spec_features(spec, feature_config, prefix=feature_name)
            statistics.append(features)
            pbar.update()
        
    if save_path is not None and save_feature_names:
        (save_path / 'feature_names').mkdir(exist_ok=True, parents=True)
        np.save(save_path / 'feature_names' / f'{feature_name}_names', names)

    statistics = np.array(statistics, dtype=np.float32)
    if save_path is not None:
        (save_path / feature_type).mkdir(exist_ok=True, parents=True)
        np.save(save_path / f'{feature_type}/{feature_name}', statistics)

    return statistics


if __name__ == '__main__':
    root_dir = Path(__file__).parent
    file_path = root_dir / 'tests' / 'LA_T_1000137.flac'
    config_path = root_dir / 'configs'
    with open(config_path / 'spectral_features.json', 'r') as config:
        spectral_features_config = json.load(config)

    paths = [file_path]
    feature_type = 'train'
    for feature_name, feature_config in spectral_features_config.items():
        statistics = calc_spec_statistics(paths=paths,
                                          feature_name=feature_name,
                                          feature_config=feature_config,
                                          feature_type=feature_type)
        statistics_test = np.load(root_dir / 'tests' / 'spectral_features' / f'LA_T_1000137_{feature_name}.npy')
        assert np.all(statistics == statistics_test), f'Test for {feature_name} not passed'
        print(feature_name, statistics.shape)
        
    print('OK')
