from pathlib import Path


class Config:
    # number of processes used for parallel computations
    num_proc = 8
    n_estimators = 5000
    max_depth = 3
    l2_reg = 1e-3
    save_models = True
    merge_train_dev = True
    use_gpu = False
    gpu_device_id = '0'

    dataset_path = Path('/media/ssd1/ASV_datasets')
    protocols_paths = {
        'ASVspoof19_LA_train': 'ASVspoof19/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
        'ASVspoof19_LA_eval': 'ASVspoof19/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',
        'ASVspoof19_LA_dev': 'ASVspoof19/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
    }
    asv_config = {
        'url': 'http://datashare.is.ed.ac.uk/download/DS_10283_3336.zip',
        'zip_name': 'DS_10283_3336.zip',
        'compressed_data': ['LA.zip',
                            'PA.zip']
    }
