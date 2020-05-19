import numpy as np
import soundfile as sf
from scipy.stats import mode, iqr, sem, skew, kurtosis
from scipy.stats import median_absolute_deviation as mad
from librosa.feature import spectral_bandwidth,\
                            spectral_rolloff,\
                            spectral_centroid,\
                            spectral_contrast,\
                            spectral_flatness

spectral_features = {
    'rollof': spectral_rolloff,
    'centroid': spectral_centroid,
    'contrast': spectral_contrast,
    'bandwidth': spectral_bandwidth,
    'flatness': spectral_flatness
}

stat_features = {
    'min': np.min,
    'max': np.max,
    'mean': np.mean,
    'std': np.std,
    'skew': skew,
    'kurtosis': kurtosis,
    'iqr': iqr,
    'sem': sem,
    'mad': mad,
    'mode': mode
}


def read_audio_file(path: str, norm_const: int = 2**15):
    """
    Read audio signal by path

    Parameters:
    path: str - path to the file
    norm_const: int - normalization constant
    
    Returns:
    signal: np.ndarray - audio signal from {-1 to 1}
    """
    
    signal, _ = sf.read(path, dtype='int16')
    signal = signal / norm_const
    return signal


class SignalLoader:
    """
    Wrapper class for loading and preprocessing of audio signal.
    Apply defined function for the loaded waveform.

    Parameters:
    function - statistics or spectral feature that should be extracted from the input signal
    abs_val: bool - take absolute value of feature or not
    get_first: bool - take only first of returned arguments or not
    normalize: bool - normalize input signal or not
    config: dict - kwargs arguments
    """
    def __init__(self,
                 function,
                 abs_val: bool = False,
                 get_first: bool = False,
                 normalize: bool = False,
                 **config):
        self.function = function
        self.abs_val = abs_val
        self.get_first = get_first
        self.normalize = normalize
        self.config = config

    def __call__(self, path: str, eps: float = 1e-8):
        try:
            signal = read_audio_file(path)
        except:
            print(f'Error occured on: {path}')
        
        if self.normalize:
            signal = (signal - signal.mean()) / signal.std()
            # signal = (signal - signal.mean()) / (signal.std() + eps)

        result = self.function(signal, **self.config)
        if self.get_first:
            result = result[0]
        if self.abs_val:
            result = np.abs(result)
            
        return result


def extract_spec_features(spectrogram: np.ndarray, config: dict, prefix: str = ''):
    """
    Extracting spectal features from the input spectrogram

    Parameters:
    spectrogram: np.ndarray - input spectrum
    config: dict - stat. features that should be extracted
    prefix: str - prefix used for naming

    Returns:
    features: np.ndarray - extracted stat. features
    feature_names: list - names of extracted features
    """

    features = []
    feature_names = []

    for spec_feature_name, stat_config in config.items():
        if not stat_config['included']:
            continue

        spec_feature = spectral_features[spec_feature_name](S=spectrogram)[0]
        for stat_feature_key, included in stat_config['stat_features'].items():
            if not included:
                continue
            
            stat_feature_name = stat_feature_key.split('_')[0]
            feature = stat_features[stat_feature_name](spec_feature)
            if stat_feature_key == 'mode_val':
                feature = feature.mode[0]
            elif stat_feature_key == 'mode_cnt':
                feature = feature.count[0]

            features.append(feature)

            name = f'{prefix}_{spec_feature_name}_{stat_feature_key}'
            feature_names.append(name)

    features = np.nan_to_num(features)
    return features, feature_names
