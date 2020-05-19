import io
import zipfile
import tarfile
from pathlib import Path, PosixPath

import requests
import numpy as np
from tqdm import tqdm


def get_stat_features(path: PosixPath,
                      included_features: dict,
                      verbose: bool = True):
    """
    Load precomputed statistical features

    Parameters:
    path: PosixPath - path to the precomputed statistical features
    included_features: dict - dictionary of used features 
    verbose: bool - display logs or not

    Returns:
    x_train: np.ndarray - training stat. features
    x_dev: np.ndarray - development stat. features
    x_val: np.ndarray - evaluation stat. features
    """

    x_train, x_dev, x_val = [], [], []
    for name, include in included_features.items():
        if not include:
            continue

        for feature_type, container in zip(['train', 'dev', 'val'], [x_train, x_dev, x_val]):
            if (path / feature_type / f'{name}.npy').exists():
                container.append(np.load(path / feature_type / f'{name}.npy'))

    x_train = np.hstack(x_train)
    x_dev = np.hstack(x_dev)
    x_val = np.hstack(x_val)
    assert x_train.shape[1] == x_dev.shape[1] == x_val.shape[1], "Number of features doesn't match"

    x_train = np.nan_to_num(x_train)
    x_dev = np.nan_to_num(x_dev)
    x_val = np.nan_to_num(x_val)

    if verbose:
        print(f'Train shape: {x_train.shape}\n'
              f'Dev shape: {x_dev.shape}\n'
              f'Eval shape: {x_val.shape}')

    return x_train, x_dev, x_val


def read_protocol_asv19(dataset_path: PosixPath,
                        path: str,
                        category: str,
                        access_type: str = 'LA',
                        dataset_name: str = 'ASVspoof19'):
    """
    Read path to audio files from the protocols

    Parameters:
    dataset_path: PosixPath - path to the dataset
    path: str - path to the protocol
    category: str - type of category (train, dev or eval)
    access_type: str - type of access (logical or physical)
    dataset_name: str - name of dataset directory

    Returns:
    file_paths: np.ndarray - paths to audio files
    labels: np.ndarray - labels of audio files
    """

    assert category in ('train', 'dev', 'eval')
    assert access_type in ('LA', 'PA')

    idx_map = {'file_name': 1, 'file_type': -1}
    with open(dataset_path / path) as f:
        protocol = f.readlines()

    # split each row to columns
    protocol = np.array(list(map(str.split, protocol)))
    labels = np.where(protocol[:, idx_map['file_type']] == 'spoof', 1, 0)

    # build file paths
    paths = np.array([str(dataset_path / dataset_name / access_type /
                          f'ASVspoof2019_{access_type}_{category}' / 'flac')] * len(protocol))
    file_names = protocol[:, idx_map['file_name']]

    file_paths = np.column_stack((paths, file_names))
    file_paths = list(map(lambda row: '/'.join(row) + '.flac', file_paths))

    return np.array(file_paths), labels


def get_data(dataset_path: PosixPath, protocol_paths: dict):
    """
    Load paths and labels of audio files according to the ASVspoof 2019 LA protocols

    Parameters:
    dataset_path: PosixPath - path to the dataset
    protocol_paths: dict - paths to ASVspoof 2019 LA protocols

    Returns:
    train_x: np.ndarray - paths to the training audio files
    train_y: np.ndarray - labels of training part
    dev_x: np.ndarray - paths to the development audio files
    dev_y: np.ndarray - labels of development part
    val_x: np.ndarray - paths to the validation audio files
    val_y: np.ndarray - labels of validation part
    """
    
    train_x, train_y = [], []
    dev_x, dev_y = [], []
    val_x, val_y = [], []
    print('Loading ASVspoof 2019 LA testing protocols...')

    for protocol, path in tqdm(protocol_paths.items()):
        _, *category = protocol.split('_')
        access_type, category, = category
        file_paths, labels = read_protocol_asv19(dataset_path, path, category, access_type)

        if category == 'train':
            train_x.extend(file_paths)
            train_y.extend(labels)
        elif category == 'dev':
            dev_x.extend(file_paths)
            dev_y.extend(labels)
        else:
            val_x.extend(file_paths)
            val_y.extend(labels)

    return (np.array(train_x), np.array(train_y)),\
           (np.array(dev_x), np.array(dev_y)),\
           (np.array(val_x), np.array(val_y))


def asv_spoof(path: PosixPath,
              url: str,
              zip_name: str,
              compressed_data: list,
              dataset_name: str = 'ASVspoof19'):
    """
    Downloading and unpacking of the ASVspoof 2019 LA dataset to the specified directory

    Parameters:
    path: PosixPath - path for unpacking of the dataset
    url: str - url for downloading of the dataset
    zip_name: str - name of the zipped dataset
    compressed_data: list - list of compressed data
    dataset_name: str - name of directory for unpacking of the dataset
    """

    path = path / dataset_name
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

        request = requests.get(url, stream=True)
        chunk_size = int(1e6)  # 1 MB
        total_size = int(request.headers.get('content-length', 0)) // chunk_size

        with open(path / zip_name, 'wb') as f:
            print(f'Loading {dataset_name} dataset...')
            for block in tqdm(request.iter_content(chunk_size=chunk_size), total=total_size):
                f.write(block)
        print('Loading completed!')

    print('Extracting data...')
    with open(path / zip_name, 'rb') as f:
        with zipfile.ZipFile(io.BytesIO(f.read())) as zf:
            zf.extractall(path=path)
    (path / zip_name).rmdir()

    for name in tqdm(compressed_data):
        if name.split('.')[-1] == 'zip':
            with zipfile.ZipFile(path / name) as f:
                f.extractall(path=path)
        else:
            with tarfile.open(path / name) as f:
                f.extractall(path=path)
        (path / name).rmdir()

    print('Extraction completed...')
