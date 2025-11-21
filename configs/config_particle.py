"""Configuration file for training NPE on particle datasets."""

from ml_collections import ConfigDict


def get_config():
    """Get the default configuration for training NPE on particle data."""
    config = ConfigDict()

    ### Experiment name and directories
    config.workdir = '/mnt/ceph/users/tnguyen/stream/trained-models'
    config.name = 'present-6D-sf5'
    config.overwrite = True
    config.checkpoint = None
    config.enable_progress_bar = True

    ### Seed configuration
    config.seed_data = 3231
    config.seed_training = 871231

    ### DataLoader configuration
    config.train_frac = 0.8
    config.train_batch_size = 64
    config.eval_batch_size = 64
    config.num_workers = 0

    ### Data configuration
    config.data = ConfigDict()
    config.data.data_type = 'preprocessed'
    config.data.root = '/mnt/ceph/users/tnguyen/stream/preprocessed_datasets/particles/'
    config.data.name = 'present-6D-sf5'
    config.data.labels = ['log_M_sat', 'log_rs_sat', 'vz', 'vphi', 'r', 'phi']
    config.data.features = ['phi1', 'phi2', 'pm1', 'pm2', 'vr', 'dist']
    config.data.num_datasets = 35
    config.data.start_dataset = 0
    config.data.num_subsamples = 5
    config.data.num_per_subsample = 100
    config.data.phi1_min = -20
    config.data.phi1_max = 10
    config.data.uncertainty_model = 'present'

    ### Model configuration
    config.model = model = ConfigDict()

    # Embedding network configuration
    # GNN -> MLP
    model.embedding_type = 'gnn'
    model.embedding_args = ConfigDict()
    model.embedding_args.gnn_args = ConfigDict()
    model.embedding_args.gnn_args.input_size = 6
    model.embedding_args.gnn_args.hidden_sizes = [32, ] * 3
    model.embedding_args.gnn_args.projection_size = 32
    model.embedding_args.gnn_args.graph_layer = "GATConv"
    model.embedding_args.gnn_args.graph_layer_params = ConfigDict()
    model.embedding_args.gnn_args.graph_layer_params.heads = 2
    model.embedding_args.gnn_args.graph_layer_params.concat = False
    model.embedding_args.gnn_args.activation_name = "leaky_relu"
    model.embedding_args.gnn_args.activation_args = None
    model.embedding_args.gnn_args.pooling = "mean"
    model.embedding_args.gnn_args.layer_norm = True
    model.embedding_args.gnn_args.norm_first = False
    model.embedding_args.mlp_args = ConfigDict()
    model.embedding_args.mlp_args.activation_name = 'gelu'
    model.embedding_args.mlp_args.activation_args = None
    model.embedding_args.mlp_args.hidden_sizes = [32, ] * 3
    model.embedding_args.mlp_args.output_size = 32
    model.embedding_args.mlp_args.batch_norm = True
    model.embedding_args.mlp_args.dropout = 0.1

    # Flow configuration
    model.flows_args = ConfigDict()
    model.flows_args.features = len(config.data.labels)
    model.flows_args.context_size = model.embedding_args.mlp_args.output_size
    model.flows_args.hidden_sizes = [64, ] * 3
    model.flows_args.num_transforms = 6
    model.flows_args.num_bins = 8
    model.flows_args.activation = 'gelu'
    model.flows_args.activation_args = None

    # Transformation configuration
    # batch transformation before the training loop, different for different embedding types
    model.batch_prep_args = ConfigDict()
    model.batch_prep_args.k = 20
    model.batch_prep_args.loop = True

    ### Optimizer and scheduler configuration
    config.optimizer = ConfigDict()
    config.optimizer.name = 'AdamW'
    config.optimizer.lr = 5e-4
    config.optimizer.betas = (0.9, 0.999)
    config.optimizer.weight_decay = 0.01
    config.scheduler = ConfigDict()
    config.scheduler.name = 'WarmUpCosineAnnealingLR'
    config.scheduler.decay_steps = 1_000_000
    config.scheduler.warmup_steps = int(0.05 * config.scheduler.decay_steps)
    config.scheduler.eta_min = 1e-5
    config.scheduler.interval = 'step'

    ### Training callbacks and configuration
    config.num_steps = config.scheduler.decay_steps
    config.patience = 20
    config.gradient_clip_val = 1.0
    config.accelerator = 'gpu'
    config.monitor = 'val_loss'
    config.mode = 'min'
    config.save_top_k = 3
    config.save_last_k = 3

    return config
