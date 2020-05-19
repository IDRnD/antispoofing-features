import json
import numpy as np
from pathlib import Path

from config import Config
from bispectrum import calc_bispectrum
from tools.dataset_loader import get_data
from tools.feature_extractor import SignalLoader
from spectral_features import calc_spec_statistics
from signal_features import calc_statistics,\
                            get_repeats,\
                            get_signal_stats,\
                            get_symmetry_diff,\
                            get_window_stats,\
                            get_hurst_exp


if __name__ == '__main__':
    root_dir = Path(__file__).parent
    config_path = root_dir / 'configs'

    save_path = root_dir / 'data'
    save_path.mkdir(exist_ok=True, parents=True)

    (train_x, train_y), (dev_x, dev_y), (val_x, val_y) = get_data(dataset_path=Config.dataset_path,
                                                                  protocol_paths=Config.protocols_paths)
    
    with open(config_path / 'signal_features.json', 'r') as config:
        signal_features_config = json.load(config)

    features_config = {
        'repeats': {'stat_func': SignalLoader(get_repeats, normalize=True, **{'num_features': 16}), 'dtype': np.uint8},
        'stats': {'stat_func': SignalLoader(get_signal_stats, normalize=True, **{'signal_features_config': signal_features_config})},
        'symmetry': {'stat_func': SignalLoader(get_symmetry_diff)},
        'wind_stats': {'stat_func': SignalLoader(get_window_stats, normalize=True)},
        'hurst': {'stat_func': SignalLoader(get_hurst_exp)}
    }

    for feature_name, config in features_config.items():
        for paths, feature_type in zip([train_x, dev_x, val_x], ['train', 'dev', 'val']):
            print(f'Calculating {feature_name} {feature_type}...')
            calc_statistics(paths=paths,
                            feature_name=feature_name,
                            feature_type=feature_type,
                            save_path=save_path,
                            **config)

    for paths, feature_type in zip([train_x, dev_x, val_x], ['train', 'dev', 'val']):
        print(f'Calculating bispectrum {feature_type}...')
        calc_bispectrum(paths=paths,
                        feature_type=feature_type,
                        config_path=config_path,
                        save_path=save_path)

    with open(config_path / 'spectral_features.json', 'r') as config:
        spectral_features_config = json.load(config)

    for feature_name, feature_config in spectral_features_config.items():
        for paths, feature_type in zip([train_x, dev_x, val_x], ['train', 'dev', 'val']):
            print(f'Calculating {feature_name} {feature_type}...')
            calc_spec_statistics(paths=paths,
                                 feature_name=feature_name,
                                 feature_config=feature_config,
                                 feature_type=feature_type,
                                 save_path=save_path)
