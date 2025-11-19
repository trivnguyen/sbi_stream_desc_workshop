
import torch
import torch.nn as nn
import pytorch_lightning as pl

from sbi_stream import flows_utils, training_utils
from sbi_stream.models.transformer import TransformerEmbedding
from sbi_stream.models.gnn import GNNEmbedding

class NPE(pl.LightningModule):
    def __init__(self, model_args, optimizer_args=None, scheduler_args=None, norm_dict=None):
        """
        Parameters
        ----------
        model_args : dict
            Arguments for the model
        optimizer_args : dict, optional
            Arguments for the optimizer. Default: None
        scheduler_args : dict, optional
            Arguments for the scheduler. Default: None
        norm_dict : dict, optional
            The normalization dictionary. For bookkeeping purposes only.
            Default: None
        """
        super().__init__()
        self.model_args = model_args
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.norm_dict = norm_dict
        self.save_hyperparameters()
        self._setup_model()

    def _setup_model(self):
        """ Initialize the model components. """
        embedding_type = self.model_args.get('embedding_type')
        embedding_args = self.model_args.get('embedding_args')
        flows_args = self.model_args.get('flows_args')

        # Create the embedding model
        if embedding_type == 'transformer':
            self.embedding_model = TransformerEmbedding(
                transformer_args=embedding_args['transformer'],
                mlp_args=embedding_args['mlp']
            )
        elif embedding_type == 'gnn':
            self.embedding_model = GNNEmbedding(
                gnn_args=embedding_args['gnn'],
                mlp_args=embedding_args['mlp']
            )
        else:
            raise ValueError(f'Unknown embedding type: {embedding_type}')
        flows_context_features = self.embedding_model.output_size

        # Create the normalizing flows
        self.flows = flows_utils.build_flows(
            features=flows_args['features'],
            hidden_features=flows_args['hidden_sizes'],
            context_features=flows_context_features,
            num_transforms=flows_args['num_transforms'],
            num_bins=flows_args['num_bins'],
            activation=flows_args.get('activation', 'relu'),
            activation_args=flows_args.get('activation_args', None),
        )

    def forward(self, batch_dict):
        return self.embedding_model(batch_dict)

    def training_step(self, batch, batch_idx):
        batch_dict = training_utils.prepare_batch(
            batch, self.model_args['embedding_type'], self.device)
        context = self.forward(batch_dict)
        log_prob = self.flows(context).log_prob(batch_dict['y'])
        loss = -log_prob.mean()
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True,
            prog_bar=True, batch_size=batch_dict['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = training_utils.prepare_batch(
            batch, self.model_args['embedding_type'], self.device)
        context = self.forward(batch_dict)
        log_prob = self.flows(context).log_prob(batch_dict['y'])
        loss = -log_prob.mean()
        self.log(
            'val_loss', loss, on_step=False, on_epoch=True,
            prog_bar=True, batch_size=batch_dict['batch_size'])
        return loss

    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """
        return training_utils.configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)
