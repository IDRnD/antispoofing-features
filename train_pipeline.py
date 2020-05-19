import json
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from config import Config
from tools.tester import calc_eer
from tools.dataset_loader import get_data, get_stat_features


if __name__ == '__main__':
    root_dir = Path(__file__).parent
    config_path = root_dir / 'configs'
    save_path = root_dir / 'data'

    with open(config_path / 'included_features.json', 'r') as config:
        included_features = json.load(config)

    train_x, dev_x, val_x = get_stat_features(path=save_path,
                                              included_features=included_features)
    (_, train_y), (_, dev_y), (_, val_y) = get_data(dataset_path=Config.dataset_path,
                                                    protocol_paths=Config.protocols_paths)

    if Config.merge_train_dev:
        train_x = np.vstack((train_x, dev_x))
        train_y = np.concatenate((train_y, dev_y))
        print('Merged train set shape:', train_x.shape)

    human_weight = 1
    spoof_weight = (train_y == 0).sum() / (train_y == 1).sum()
    class_weights = [human_weight, spoof_weight]

    rf = RandomForestClassifier(n_estimators=Config.n_estimators,
                                max_depth=Config.max_depth,
                                class_weight={0: class_weights[0], 1: class_weights[1]},
                                n_jobs=Config.num_proc)

    cb = CatBoostClassifier(n_estimators=Config.n_estimators, 
                            max_depth=Config.max_depth,
                            class_weights=class_weights,
                            l2_leaf_reg=Config.l2_reg,
                            verbose=0,
                            task_type='GPU' if Config.use_gpu else 'CPU',
                            devices=Config.gpu_device_id if Config.use_gpu else None)

    lgbm = LGBMClassifier(n_estimators=Config.n_estimators,
                        max_depth=Config.max_depth,
                        class_weight={0: class_weights[0], 1: class_weights[1]},
                        reg_lambda=Config.l2_reg,
                        objective='binary',
                        device='gpu' if Config.use_gpu else 'cpu',
                        n_jobs=Config.num_proc)

    for clf, clf_name in zip([rf, cb, lgbm], ['random_forest', 'cat_boost', 'light_gbm']):
        print(f'Training {clf_name} classifier...')
        clf.fit(X=train_x, y=train_y)

        pred_proba = clf.predict_proba(val_x).T[1]
        eer, threshold = calc_eer(val_y, pred_proba)
        print(f'{clf_name}\nValidation EER: {eer}\nThreshold: {threshold}\n')

        if Config.save_models:
            (root_dir / 'models').mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, root_dir / 'models' / clf_name)
    