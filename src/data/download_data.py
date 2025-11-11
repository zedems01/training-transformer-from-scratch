import os
import urllib.request
import tarfile
from tqdm import tqdm
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download a file from URL to the specified output path with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_wmt14_en_fr(data_dir="./src/data/raw"):
    """
    Download and extract WMT14 English-French dataset.
    The dataset includes:
    - Europarl v7
    - Common Crawl corpus
    - News Commentary
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # dataset urls for French-English
    urls = {
        'europarl': 'https://www.statmt.org/europarl/v7/fr-en.tgz',
        'common_crawl': 'https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
        'news_commentary': 'https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz'
    }
    
    # download and extract each dataset
    for dataset_name, url in urls.items():
        logging.info(f"Downloading {dataset_name}...")
        
        output_file = os.path.join(data_dir, f"{dataset_name}.tgz")
        if not os.path.exists(output_file):
            download_url(url, output_file)
        
        logging.info(f"Extracting {dataset_name}...")
        with tarfile.open(output_file) as tar:
            tar.extractall(path=data_dir, filter='data')
        
        # clean up tgz file
        os.remove(output_file)
    
    logging.info(f"Download complete! Dataset is ready in {data_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download WMT14 English-French dataset')
    parser.add_argument('--data-dir', type=str, default='./src/data/raw', help='Directory to save the dataset')
    args = parser.parse_args()
    download_wmt14_en_fr(args.data_dir)