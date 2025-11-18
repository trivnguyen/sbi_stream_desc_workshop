
from typing import Optional, Callable, Dict

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import Tensor


class Transformer(nn.Module):
    """ Transformer model """

    def __init__(
        self, d_feat_in: int, d_pos_in: int, d_feat: int = 32, d_pos: int = 32, nhead: int = 4,
        num_encoder_layers: int = 4, dim_feedforward: int = 128, sum_features: bool = False,
        use_embedding: bool = True, activation_name: str, activation_args: Optional[Dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        d_feat_in : int
            The dimension of the input features (per token).
        d_pos_in : int
            The dimension of the input positional features.
        d_feat : int
            The dimension of the projected feature embedding.
        d_pos : int
            The dimension of the projected positional embedding.
        nhead : int
            The number of heads in the multihead attention modules.
        num_encoder_layers : int
            The number of sub-encoder-layers in the encoder.
        sum_features : bool, optional
            Whether to sum the features along the sequence dimension. Default: False
        dim_feedforward : int
            The dimension of the feedforward network model.
        use_embedding : bool
            Whether to use embedding layers. Default: True
        activation_fn : callable, optional
            The activation function to use for the embedding layer. Default: None
        """
        super().__init__()
        self.d_feat = d_feat
        self.d_pos = d_pos
        self.d_model = d_feat + d_pos
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.use_embedding = use_embedding
        self.activation_fn = activation_fn

        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers)
        self.feat_embedding_layer = nn.Linear(d_feat_in, d_feat)
        self.pos_embedding_layer = nn.Linear(d_pos_in, d_pos)

    def forward(self, x: Tensor, pos: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        x: (batch, seq, d_feat_in)
        pos: (batch, seq, d_pos_in)
        padding_mask: (batch, seq) boolean mask where True indicates padding
        """
        x = self.feat_embedding_layer(x)
        pos = self.pos_embedding_layer(pos)
        src = torch.cat((x, pos), dim=-1)
        output = self.transformer_encoder(
            src, src_key_padding_mask=padding_mask)

        # NOTE: dimension only works when batch_first=True
        if padding_mask is None:
            output = output.sum(dim=1)
        else:
            if not self.training:
                # apply correct padding mask for evaluation
                # this happens because with both self.eval() and torch.no_grad()
                # the transformer encoder may shorten the output length to
                # the max non-padded length in the batch
                max_seq_len = output.shape[1]
                padding_mask = padding_mask[:, :max_seq_len]
            output = output.masked_fill(padding_mask.unsqueeze(-1), 0)
            output = output.sum(dim=1)

        return output


class TransformerEmbedding(nn.Module):
    """ Transformer + MLP embedding model """

    def __init__(
        self, transformer_args: Dict, mlp_args: Dict):
        """
        Parameters
        ----------
        transformer_args : dict
            The arguments for the Transformer model.
        mlp_args : dict
            The arguments for the MLP model.
        """
        super().__init__()
        self.transformer_args = transformer_args
        self.mlp_args = mlp_args
        self.transformer = Transformer(
            d_feat_in=transformer_args['d_feat_in'],
            d_pos_in=transformer_args['d_pos_in'],
            d_feat=transformer_args.get('d_feat', 32),
            d_pos=transformer_args.get('d_pos', 32),
            nhead=transformer_args.get('nhead', 4),
            num_encoder_layers=transformer_args.get('num_encoder_layers', 4),
            dim_feedforward=transformer_args.get('dim_feedforward', 128),
            sum_features=transformer_args.get('sum_features', False),
            use_embedding=transformer_args.get('use_embedding', True),
            activation_name=transformer_args.get('activation_name', 'ReLU'),
            activation_args=transformer_args.get('activation_args', None),
        )
        self.mlp = mlp.MLPBatchNorm(
            input_size=self.gnn.hidden_sizes[-1],
            output_size=mlp_args['output_size'],
            hidden_sizes=mlp_args.get('hidden_sizes', []),
            activation_fn=utils.get_activation(
                mlp_args.get('activation_name', 'relu'),
                mlp_args.get('activation_args', {})
            ),
            batch_norm=mlp_args.get('batch_norm', False),
            dropout=mlp_args.get('dropout', 0.0)
        )



    def forward(self, x, pos, padding_mask=None):
        """
        x: (batch, seq, d_feat_in)
        pos: (batch, seq, d_pos_in)
        padding_mask: (batch, seq) boolean mask where True indicates padding
        """
        x = self.transformer(x, pos, padding_mask)
        x = self.mlp(x)
        return x