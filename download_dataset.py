from config import Config
from tools.dataset_loader import asv_spoof


if __name__ == '__main__':
    asv_spoof(path=Config.dataset_path,
              url=Config.asv_config['url'],
              zip_name=Config.asv_config['zip_name'],
              compressed_data=Config.asv_config['compressed_data'])
