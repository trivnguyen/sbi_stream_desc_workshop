
import os
import pickle
import sys
import shutil
import yaml
from tqdm import tqdm

import numpy as np
import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import matplotlib as mpl
from absl import flags, logging
from ml_collections import ConfigDict, config_flags

import datasets


def preprocess(config: ConfigDict):

    workdir = os.path.join(config.root_out, config.name_out)

    # convert config to yaml and save
    os.makedirs(workdir, exist_ok=True)
    config_dict = config.to_dict()
    config_path = os.path.join(workdir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # read in the dataset and prepare the data loader for training
    data_raw_dir = os.path.join(config.root, config.name)

    logging.info("Processing raw data from %s", data_raw_dir)
    for i in tqdm(range(config.start_dataset, config.start_dataset + config.num_datasets)):
        data = datasets.read_raw_particle_datasets(
            data_raw_dir,
            features=config.features,
            labels=config.labels,
            num_datasets=config.get('num_datasets', 1),
            start_dataset=config.get('start_dataset', 0),
            num_subsamples=config.get('num_subsamples', 1),
            num_per_subsample=config.get('num_per_subsample', None),
            phi1_min=config.get('phi1_min', None),
            phi1_max=config.get('phi1_max', None),
            uncertainty_model=config.get('uncertainty_model', None),
        )
        if data is not None:
            data_out_path = os.path.join(workdir, f'data.{i}.pkl')
            logging.info("Saving processed data to %s", data_out_path)
            with open(data_out_path, "wb") as f:
                pickle.dump(data, f)

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training or sampling hyperparameter configuration.",
        lock_config=True,
    )
    # Parse flags
    FLAGS(sys.argv)

    # Start training run
    preprocess(config=FLAGS.config)
