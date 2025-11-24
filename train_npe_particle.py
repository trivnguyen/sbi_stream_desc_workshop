
import os
import pickle
import sys
sys.path.append('/global/homes/t/tvnguyen/sbi_stream')
import shutil
import yaml

import numpy as np
import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from absl import flags, logging
from ml_collections import ConfigDict, config_flags

import datasets
from sbi_stream.npe import NPE


def generate_seeds(base_seed, num_seeds, seed_range=(0, 2**32 - 1)):
    """Generate a list of RNG seeds deterministically from a base seed."""
    np.random.seed(base_seed)
    return np.random.randint(seed_range[0], seed_range[1], size=num_seeds, dtype=np.uint32)


def train(config: ConfigDict):

    # set up work directory
    name = config.get('name', 'default_particle')
    logging.info("Starting training run {} at {}".format(name, config.workdir))

    workdir = os.path.join(config.workdir, name)

    checkpoint_path = None
    if config.get('checkpoint') is not None:
        if os.path.isabs(config.checkpoint):
            checkpoint_path = config.checkpoint   # use full path
        else:
            checkpoint_path = os.path.join(
                workdir, 'lightning_logs/checkpoints', config.checkpoint)
        if not os.path.exists(checkpoint_path):
            raise ValueError(f'Checkpoint {checkpoint_path} does not exist')
        logging.info(f'Reading checkpoint {checkpoint_path}')

    if os.path.exists(workdir):
        if checkpoint_path is None:
            if config.overwrite:
                shutil.rmtree(workdir)
            else:
                raise ValueError(
                    f"Workdir {workdir} already exists. Please set overwrite=True "\
                        "to overwrite the existing directory.")

    # convert config to yaml and save
    os.makedirs(workdir, exist_ok=True)
    config_dict = config.to_dict()
    config_path = os.path.join(workdir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # generate data and training seed
    data_seed = generate_seeds(config.seed_data, 10_000)[0]
    training_seed = generate_seeds(config.seed_training, 10_000)[0]

    # read in the dataset and prepare the data loader for training
    if config.data.data_type == 'raw':
        dataset = datasets.read_raw_particle_datasets(
            os.path.join(config.data.root, config.data.name),
            features=config.data.features,
            labels=config.data.labels,
            num_datasets=config.data.get('num_datasets', 1),
            start_dataset=config.data.get('start_dataset', 0),
            num_subsamples=config.data.get('num_subsamples', 1),
            num_per_subsample=config.data.get('num_per_subsample', None),
            phi1_min=config.data.get('phi1_min', None),
            phi1_max=config.data.get('phi1_max', None),
            uncertainty_model=config.data.get('uncertainty_model', None),
        )
    elif config.data.data_type == 'preprocessed':
        dataset = datasets.read_processed_particle_datasets(
            os.path.join(config.data.root, config.data.name),
            num_datasets=config.data.get('num_datasets', 1),
            start_dataset=config.data.get('start_dataset', 0),
        )
    else:
        raise ValueError(f"Unknown data_type {config.data.data_type}")

    # Prepare dataloader with the appropriate norm_dict
    # Use config value if provided, otherwise compute based on num_subsamples
    train_loader, val_loader, norm_dict = datasets.prepare_particle_dataloader(
        dataset,
        train_frac=config.train_frac,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        seed=data_seed,
        num_subsamples=config.data.get('num_subsamples', 1),
        norm_dict=None,
    )

    # Initialize the model
    model = NPE(
        model_args=config.model,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        norm_dict=norm_dict,
        freeze_components=config.get('freeze_components')
    )

    # create the trainer object
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor=config.monitor, patience=config.patience, mode=config.mode,
            verbose=True),
        pl.callbacks.ModelCheckpoint(
            monitor=config.monitor, save_top_k=config.get('save_top_k', 3),
            filename='best-{epoch}-{step}-{train_loss:.4f}-{val_loss:.4f}',
            mode=config.mode, save_weights_only=False),
        pl.callbacks.ModelCheckpoint(
            monitor='epoch', save_top_k=config.get('save_last_k', 3),
            filename='last-{epoch}-{step}-{train_loss:.4f}-{val_loss:.4f}',
            mode='max', save_weights_only=False),
        pl.callbacks.LearningRateMonitor("step"),
    ]
    train_logger = pl_loggers.TensorBoardLogger(workdir, version='')
    trainer = pl.Trainer(
        default_root_dir=workdir,
        max_steps=config.num_steps,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=train_logger,
        enable_progress_bar=config.get("enable_progress_bar", True),
        gradient_clip_val=config.get("gradient_clip_val", 0.0),
    )

    # train the model
    logging.info("Training model...")
    pl.seed_everything(training_seed)
    if (checkpoint_path is not None) and (config.get('reset_optimizer', False)):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(
            model, train_loader, val_loader, ckpt_path=checkpoint_path)


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
    train(config=FLAGS.config)
