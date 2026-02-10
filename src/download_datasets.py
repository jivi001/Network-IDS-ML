"""
Dataset downloader for NIDS project.
Downloads NSL-KDD and UNSW-NB15 datasets from public sources.
"""

import os
import requests
import zipfile
import io
import pandas as pd


# --- NSL-KDD ---
NSL_KDD_URLS = {
    'train': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt',
    'test': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt',
}

NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

# Attack type mapping for NSL-KDD
NSL_KDD_ATTACK_MAP = {
    'normal': 'Normal',
    # DoS attacks
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
    'processtable': 'DoS', 'mailbomb': 'DoS',
    # Probe attacks
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L attacks
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
    'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L',
    'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L', 'worm': 'R2L',
    # U2R attacks
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'perl': 'U2R',
    'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
}


def download_file(url: str, dest_path: str) -> bool:
    """Download a file from URL to destination path."""
    try:
        print(f"  Downloading: {url}")
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  Saved: {dest_path} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        return False


def download_nsl_kdd(data_dir: str = 'data/raw') -> bool:
    """
    Download NSL-KDD dataset and save as properly formatted CSVs.

    Args:
        data_dir: Directory to save the data

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("DOWNLOADING NSL-KDD DATASET")
    print("=" * 60)

    os.makedirs(data_dir, exist_ok=True)
    success = True

    for split, url in NSL_KDD_URLS.items():
        raw_path = os.path.join(data_dir, f'nsl_kdd_{split}_raw.txt')
        csv_path = os.path.join(data_dir, f'nsl_kdd_{split}.csv')

        # Check if already exists
        if os.path.exists(csv_path):
            print(f"  [SKIP] {csv_path} already exists")
            continue

        if not download_file(url, raw_path):
            success = False
            continue

        # Parse the raw text file into a proper CSV
        try:
            df = pd.read_csv(raw_path, header=None, names=NSL_KDD_COLUMNS)

            # Map specific attack names to categories
            df['label'] = df['label'].str.strip().str.lower()
            df['label'] = df['label'].map(NSL_KDD_ATTACK_MAP).fillna('Unknown')

            # Drop difficulty_level (not a real feature)
            df = df.drop(columns=['difficulty_level'], errors='ignore')

            df.to_csv(csv_path, index=False)
            print(f"  Processed: {csv_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
            print(f"  Class distribution:\n{df['label'].value_counts().to_string()}")

            # Clean up raw file
            if os.path.exists(raw_path):
                os.remove(raw_path)

        except Exception as e:
            print(f"  ERROR processing {raw_path}: {e}")
            success = False

    return success


def download_unsw_nb15(data_dir: str = 'data/raw') -> bool:
    """
    Download UNSW-NB15 dataset from public mirror.
    Note: The full dataset may require manual download from the official source.
    This downloads the smaller training/testing subset.
    """
    print("\n" + "=" * 60)
    print("DOWNLOADING UNSW-NB15 DATASET")
    print("=" * 60)

    UNSW_URLS = {
        'train': 'https://raw.githubusercontent.com/InitRoot/UNSW-NB15-Datasets/master/UNSW_NB15_training-set.csv',
        'test': 'https://raw.githubusercontent.com/InitRoot/UNSW-NB15-Datasets/master/UNSW_NB15_testing-set.csv',
    }

    os.makedirs(data_dir, exist_ok=True)
    success = True

    for split, url in UNSW_URLS.items():
        csv_path = os.path.join(data_dir, f'unsw_nb15_{split}.csv')

        if os.path.exists(csv_path):
            print(f"  [SKIP] {csv_path} already exists")
            continue

        if not download_file(url, csv_path):
            success = False
            continue

        try:
            df = pd.read_csv(csv_path, low_memory=False)

            # Standardize label column
            if 'attack_cat' in df.columns:
                df['label'] = df['attack_cat'].fillna('Normal').str.strip()
                df.loc[df['label'] == '', 'label'] = 'Normal'
                # Drop the original binary label and attack_cat
                df = df.drop(columns=['attack_cat'], errors='ignore')
                if 'Label' in df.columns:
                    df = df.drop(columns=['Label'], errors='ignore')

            df.to_csv(csv_path, index=False)
            print(f"  Processed: {csv_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
            print(f"  Class distribution:\n{df['label'].value_counts().to_string()}")

        except Exception as e:
            print(f"  ERROR processing {csv_path}: {e}")
            success = False

    return success


def download_all(data_dir: str = 'data/raw') -> dict:
    """Download all available datasets."""
    results = {}
    results['nsl_kdd'] = download_nsl_kdd(data_dir)
    results['unsw_nb15'] = download_unsw_nb15(data_dir)

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        icon = "✓" if status else "✗"
        print(f"  {icon} {name}: {'Success' if status else 'Failed'}")

    return results


def download_cic_ids2017(data_dir: str = 'data/raw'):
    """
    Placeholder for CIC-IDS2017 download.
    
    NOTE: CIC-IDS2017 is a massive dataset (GBs of PCAPs/CSVs).
    Automated download is often unstable due to server restrictions.
    
    Instructions:
    1. Go to https://www.unb.ca/cic/datasets/ids-2017.html
    2. Download 'MachineLearningCSV.zip'
    3. Extract to data/raw/cic_ids2017/
    """
    print("\n" + "=" * 60)
    print("CIC-IDS2017 DATASET")
    print("=" * 60)
    print("  [INFO] Automated download not implemented for CIC-IDS2017 due to size.")
    print(f"  Please manually download and extract to: {os.path.join(data_dir, 'cic_ids2017')}")
    return False



if __name__ == '__main__':
    download_all()
