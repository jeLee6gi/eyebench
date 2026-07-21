import argparse
from pathlib import Path

import pymovements as pm
import rdata
import requests
from loguru import logger
from pymovements import ResourceDefinitions
from tqdm import tqdm

from src.configs.constants import DataSets

logger.add('logs/preprocessing.log', level='INFO')

BASE_OSF_URL = 'https://osf.io/download/'
AUXILIARY_FILES: dict[str, dict[str, str]] = {
    DataSets.MECO_L2: {  # Hosted on MECO L2: The Multilingual Eye-movement COrpus, L2 (English) - https://osf.io/q9h43
        'MECOL2W1/demographics/joint.ind.diff.l2.rda': '4zu8d',
        'MECOL2W2/demographics/joint.ind.diff.l2.w2.rda': 'keuvm',
        'MECOL2/stimuli/texts.meco.l2.rda': 'zwfdb',
    },
}


def download_auxiliary_files(root: Path, dataset_name: str) -> None:
    """Download auxiliary resources not covered by DatasetLibrary for a specific dataset."""
    if dataset_name not in AUXILIARY_FILES:
        return

    for relative_path, resource_id in AUXILIARY_FILES[dataset_name].items():
        destination = root / relative_path
        if destination.exists():
            logger.info(
                f'{relative_path} already present at {destination}. Continuing...'
            )
            continue

        url = f'{BASE_OSF_URL}{resource_id}'
        logger.info(f'Downloading {relative_path} from {url}')
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, 'wb') as fp:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    fp.write(chunk)


def convert_rda_to_csv(root: Path, dataset_name: str) -> None:
    """Convert RDA files to CSV for specific datasets."""
    if dataset_name != DataSets.MECO_L2:
        return
    rda_path = root / 'MECOL2/stimuli/texts.meco.l2.rda'
    csv_path = root / 'MECOL2/stimuli/stimuli.csv'

    if csv_path.exists():
        logger.info(f'{csv_path} already exists. Skipping conversion...')
        return

    if not rda_path.exists():
        logger.warning(f'{rda_path} not found. Skipping conversion...')
        return

    logger.info(f'Converting {rda_path} to {csv_path}')
    rda_data = rdata.read_rda(rda_path)
    df = rda_data['d']
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f'Saved stimuli CSV to {csv_path}')


# ROAMM is not registered in pymovements.DatasetLibrary and is distributed as
# a full OSF *project* (folder tree of per-subject .pkl files, wiki_stories
# stimulus/AOI files, and label files), not a small set of individually
# addressable files like the MECOL2 auxiliary resources above. We mirror the
# entire OSF storage tree with the public OSF API rather than using pm.Dataset.
ROAMM_OSF_PROJECT_ID = 'kmvgb'


def _download_osf_folder(api_url: str, dest_dir: Path) -> None:
    """Recursively mirror an OSF storage folder to a local directory.

    OSF's file-listing endpoint paginates (default page size is small
    relative to subject_ml_data's 40+ subject folders), so a naive single
    request only sees the first page. We must follow links['next'] until
    it's null or we silently drop most of the dataset.
    """
    next_url: str | None = api_url
    while next_url is not None:
        response = requests.get(next_url, timeout=60)
        response.raise_for_status()
        payload = response.json()

        for entry in payload['data']:
            attrs = entry['attributes']
            if attrs['kind'] == 'folder':
                _download_osf_folder(
                    entry['relationships']['files']['links']['related']['href'],
                    dest_dir / attrs['name'],
                )
            else:
                dest_path = dest_dir / attrs['name']
                if dest_path.exists():
                    continue
                dest_dir.mkdir(parents=True, exist_ok=True)
                download_url = entry['links']['download']
                logger.info(f'Downloading {dest_path}')
                with requests.get(download_url, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(dest_path, 'wb') as fp:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            fp.write(chunk)

        next_url = payload.get('links', {}).get('next')


def download_roamm(root: Path) -> None:
    """Download the ROAMM dataset (subject_ml_data, wiki_stories, labels) from OSF."""
    dest_root = root / DataSets.ROAMM
    if dest_root.exists() and any(dest_root.iterdir()):
        logger.info(f'ROAMM already present at {dest_root}. Continuing...')
        return

    logger.info(f'Downloading ROAMM from OSF project {ROAMM_OSF_PROJECT_ID}...')
    dest_root.mkdir(parents=True, exist_ok=True)
    api_url = f'https://api.osf.io/v2/nodes/{ROAMM_OSF_PROJECT_ID}/files/osfstorage/'
    _download_osf_folder(api_url, dest_root)


def prepare_dataset_definition(dataset_name: str):
    """Prepare dataset definition with gaze files disabled."""
    dataset_def = pm.DatasetLibrary.get(dataset_name)
    dataset_def.resources = ResourceDefinitions(
        [resource for resource in dataset_def.resources if resource.content != 'gaze']
    )

    return dataset_def


def load_or_download_dataset(
    dataset_name: str, data_path: Path, download: bool = False
) -> None:
    """Load or download a dataset based on the flag."""
    if dataset_name == DataSets.MECO_L2:
        dataset_def_w1 = prepare_dataset_definition(f'{dataset_name}W1')
        dataset_def_w2 = prepare_dataset_definition(f'{dataset_name}W2')
        dataset_w1 = pm.Dataset(dataset_def_w1, data_path / DataSets.MECO_L2W1)
        dataset_w2 = pm.Dataset(dataset_def_w2, data_path / DataSets.MECO_L2W2)
        if download:
            dataset_w1.download()
            dataset_w2.download()
        else:
            dataset_w1.load()
            dataset_w2.load()
    else:
        dataset_def = prepare_dataset_definition(dataset_name)
        dataset = pm.Dataset(dataset_def, data_path / dataset_name)
        if download:
            dataset.download()
        else:
            dataset.load()


def main() -> int:
    data_path = Path('data')
    data_path.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='')
    args = parser.parse_args()

    dataset = args.dataset

    if dataset:
        datasets_names = dataset.split(',')
    else:
        datasets_names = [
            DataSets.ONESTOP,
            DataSets.COPCO,
            DataSets.POTEC,
            DataSets.SBSAT,
            DataSets.HALLUCINATION,
            DataSets.MECO_L2,
        ]

    for dataset_name in tqdm(
        datasets_names,
        desc='Downloading datasets',
        unit='dataset',
        total=len(datasets_names),
    ):
        if dataset_name == DataSets.ROAMM:
            # Not in pymovements.DatasetLibrary and not included in the
            # default (no --dataset) run given its size (46M+ samples);
            # must be requested explicitly via --dataset ROAMM.
            download_roamm(data_path)
            continue

        try:
            load_or_download_dataset(dataset_name, data_path, download=False)
            logger.info(f'{dataset_name} already downloaded. Continuing...')
        except Exception:
            logger.info(f'{dataset_name} not downloaded yet. Downloading...')
            load_or_download_dataset(dataset_name, data_path, download=True)

        download_auxiliary_files(data_path, dataset_name)
        convert_rda_to_csv(data_path, dataset_name)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
