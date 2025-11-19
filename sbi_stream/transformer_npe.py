
import torch
import torch.nn as nn
import pytorch_lightning as pl

from sbi_stream import flows_utils
from sbi_stream import models_utils
from sbi_stream.models.transformer import Transformer
from sbi_stream.models.mlp import MLPBatchNorm


class TransformerNPE(pl.LightningModule):
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
        embedding_args = self.model_args.get('embedding_args')
        flows_args = self.model_args.get('flows_args')

        # Create the embedding model
        self.transformer = Transformer(
            feat_input_size=transformer_args['feat_input_size'],
            pos_input_size=transformer_args['pos_input_size'],
            feat_embed_size=transformer_args.get('feat_embed_size', 32),
            pos_embed_size=transformer_args.get('pos_embed_size', 32),
            nhead=transformer_args.get('nhead', 4),
            num_encoder_layers=transformer_args.get('num_encoder_layers', 4),
            dim_feedforward=transformer_args.get('dim_feedforward', 128),
            sum_features=transformer_args.get('sum_features', False),
            use_embedding=transformer_args.get('use_embedding', True),
            activation_name=transformer_args.get('activation_name', 'ReLU'),
            activation_args=transformer_args.get('activation_args', None),
        )
        self.mlp = MLPBatchNorm(
            input_size=self.transformer.d_model,
            output_size=mlp_args['output_size'],
            hidden_sizes=mlp_args.get('hidden_sizes', []),
            activation_fn=utils.get_activation(
                mlp_args.get('activation_name', 'relu'),
                mlp_args.get('activation_args', {})
            ),
            batch_norm=mlp_args.get('batch_norm', False),
            dropout=mlp_args.get('dropout', 0.0)
        )
        flows_context_features = mlp_args['output_size']

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

    def forward(self, x, pos, padding_mask=None):
        """
        x: (batch, seq, feat_input_size)
        pos: (batch, seq, pos_input_size)
        padding_mask: (batch, seq) boolean mask where True indicates padding
        """
        x = self.transformer(x, pos, padding_mask)
        x = self.mlp(x)
        return x

    def prepare_batch(batch):
        """ Prepare batch for transformer embedding. """
        x, theta, pos, padding_mask = batch
        return {
            'x': x.to(self.device),
            'theta': theta.to(self.device),
            'pos': pos.to(self.device),
            'padding_mask': padding_mask.to(self.device),
            'batch_size': x.size(0),
        }

    def training_step(self, batch, batch_idx):
        batch_dict = self.prepare_batch(batch)
        context = self.forward(
            x=batch_dict['x'],
            pos=batch_dict['pos'],
            padding_mask=batch_dict['padding_mask']
        )
        log_prob = self.flows(context).log_prob(batch_dict['theta'])
        loss = -log_prob.mean()
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True,
            prog_bar=True, batch_size=batch_dict['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = self.prepare_batch(batch)
        context = self.forward(
            x=batch_dict['x'],
            pos=batch_dict['pos'],
            padding_mask=batch_dict['padding_mask']
        )
        log_prob = self.flows(context).log_prob(batch_dict['theta'])
        loss = -log_prob.mean()
        self.log(
            'val_loss', loss, on_step=False, on_epoch=True,
            prog_bar=True, batch_size=batch_dict['batch_size'])
        return loss

    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """
        return models_utils.configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)
