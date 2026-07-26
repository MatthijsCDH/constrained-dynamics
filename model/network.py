import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_autotune_level=4 "
    "--xla_gpu_triton_gemm_any=true"
)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache", "network")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = _CACHE_DIR

import jax
import jax.numpy as jnp
from jax import random, value_and_grad, lax
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)

import time
import sys
import gc

from functools import partial
from typing import NamedTuple, Optional, Tuple, Callable, Dict, Any
from abc import ABC, abstractmethod
from functools import partial
import numpy as np
from math import prod
import optax
import flax
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from configs.configurations import (
    MLPTrainConfig, PINNTrainConfig, HNNTrainConfig, LNNTrainConfig, InferenceConfig, BenchmarkConfig,
)
from model.physics_loss import (
    PhysicsLoss, MLPLoss, HNNLoss, LNNLoss,
)


# ── Layer type IDs ────────────────────────────────────────────────────────────
FLATTEN                 = 0
FC                      = 1
CONV                    = 2
POOL_MAX                = 3
POOL_SUM                = 4
LAYER_NORM              = 5
TRANSFORMER             = 6
DROPOUT                 = 7
ADD                     = 8
GPOOL_MAX               = 9
GPOOL_SUM               = 10
NNUPSAMPLING            = 11
CONCATENATION           = 12
BUPSAMPLING             = 13
BIAS                    = 14
FLATTEN_SPATIAL         = 15
UNFLATTEN_SPATIAL       = 16
POSITIONALEMBEDDING2D   = 17
TRANSFORMERENCODER      = 18
BRANCH                  = 19
CROSS_ATTENTION         = 20

# ── Params and State ──────────────────────────────────────────────────────────
class FCParams(NamedTuple):
    W: jnp.ndarray
    b: jnp.ndarray

class ConvParams(NamedTuple):
    W: jnp.ndarray
    b: jnp.ndarray

class NormParams(NamedTuple):
    gamma: jnp.ndarray
    beta:  jnp.ndarray

class TransformerParams(NamedTuple):
    Wq: jnp.ndarray
    Wk: jnp.ndarray
    Wv: jnp.ndarray
    Wo: jnp.ndarray

class BiasParams(NamedTuple):
    b: jnp.ndarray 

class PositionalEmbedding2DParams(NamedTuple):
    E_row: jnp.ndarray 
    E_col: jnp.ndarray 

class TransformerEncoderParams(NamedTuple):
    attn:  TransformerParams
    bo:    BiasParams
    norm1: NormParams
    fc1:  FCParams
    fc2:  FCParams
    norm2: NormParams

class CrossAttentionParams(NamedTuple):
    Wq: jnp.ndarray
    Wk: jnp.ndarray
    Wv: jnp.ndarray
    Wo: jnp.ndarray


class TrainState(NamedTuple):
    params:    any
    opt_state: any
    rng:       jax.random.PRNGKey

# ── Layer configs ─────────────────────────────────────────────────────────────
class FCLayerConfig(NamedTuple):
    activation: Optional[str]
    units:      Optional[int]
    type:       int = FC

class ConvLayerConfig(NamedTuple):
    strides:      Tuple[int, int]
    padding:      any
    activation:   Optional[str]
    rhs_dilation: Tuple[int, int]
    lhs_dilation: Tuple[int, int]
    b_shape:      Tuple[int, ...]
    type:         int = CONV

class PoolMaxLayerConfig(NamedTuple):
    window:  Tuple[int, int, int, int]
    strides: Tuple[int, int, int, int]
    type:    int = POOL_MAX

class PoolSumLayerConfig(NamedTuple):
    window:  Tuple[int, int, int, int]
    strides: Tuple[int, int, int, int]
    type:    int = POOL_SUM

class LayerNormConfig(NamedTuple):
    epsilon: float = 1e-5
    type:    int   = LAYER_NORM

class FlattenConfig(NamedTuple):
    type: int = FLATTEN

class DropoutConfig(NamedTuple):
    p:    float
    type: int = DROPOUT

class AddConfig(NamedTuple):
    skip: int
    type: int = ADD

class GPoolMaxConfig(NamedTuple):
    ndim: int
    type: int = GPOOL_MAX

class GPoolSumConfig(NamedTuple):
    ndim: int
    type: int = GPOOL_SUM

class NearestNeighbourUpsamplingConfig(NamedTuple):
    scaling: int
    ndim:    int
    type:    int = NNUPSAMPLING

class ConcatenatingConfig(NamedTuple):
    skip: int
    type: int = CONCATENATION

class BilinearUpsamplingConfig(NamedTuple):
    scaling: int
    ndim:    int
    new_spatial: Tuple[int, ...]
    type:    int = BUPSAMPLING

class TransformerConfig(NamedTuple):
    heads:  int
    d_head: int
    scale:  float
    p_dropout: float
    type:   int = TRANSFORMER

class BiasConfig(NamedTuple):
    b_shape: Tuple[int, ...]
    type:   int = BIAS

class FlattenSpatialConfig(NamedTuple):
    n_tokens: int
    type:   int = FLATTEN_SPATIAL

class UnFlattenSpatialConfig(NamedTuple):
    shape:  Tuple[int, ...]
    type:   int = UNFLATTEN_SPATIAL

class PositionalEmbedding2DConfig(NamedTuple):
    height: int
    width:  int
    type:   int = POSITIONALEMBEDDING2D

class TransformerEncoderConfig(NamedTuple):
    heads:  int
    d_head: int
    d_ff:   int
    scale:  float
    p_dropout_atten: float
    p_dropout_fc: float
    type:   int = TRANSFORMERENCODER

class BranchConfig(NamedTuple):
    skip: int
    type: int = BRANCH

class CrossAttentionConfig(NamedTuple):
    heads:           int
    d_head:          int
    scale:           float
    skip:            int
    p_dropout:       float
    type:            int = CROSS_ATTENTION


# ── Neural Network ────────────────────────────────────────────────────────────
class NeuralNetwork:
    """
    A fully custom convolutional neural network (CNN) implementation built using JAX.
    This class provides a framework for defining, training and evaluating CNN architectures
    using JAX operations. It supports multiple layer types including convolution, pooling,
    flatten, fully connected, and upsampling. Users may specify an arbitrary layer sequence
    through defining an architecture in the following manner:

    architecture = [
        {"type": "fc", "units": 128, "activation": "relu"},
        {"type": "flatten"},
        {"type": "layer_normalization"},
        {"type": "conv", "filters": 64, "kernel_size": 3, "stride": 1, "padding": 1,
         "lhs_dilation": (1,1), "rhs_dilation": (1,1), "activation": "leaky_relu"},
        {"type": "pool_max", "kernel_size": 2, "stride": 2},
        {"type": "pool_sum", "kernel_size": 2, "stride": 2},
        {"type": "transformer", "heads": 8},
        {"type": "add", "skip": 0},           # sums output of nth layer with current
        {"type": "gpool_max"},
        {"type": "gpool_sum"},
        {"type": "nearest_neighbour_upsampling", "scaling": 2},
        {"type": "concatenation", "skip": 0},  # concatenates output of nth layer with current
        {"type": "bilinear_upsampling", "scaling": 2},
    ]
    """

    def __init__(self, config, X=None, y=None):
        self.t0_init = time.time()

        if isinstance(config, BenchmarkConfig):
            raise NotImplementedError()
        
        elif isinstance(config, InferenceConfig):
            raise NotImplementedError()

        elif isinstance(config, (MLPTrainConfig, PINNTrainConfig, HNNTrainConfig, LNNTrainConfig)):
            self.network_config   = config.network
            self.train_config     = config.training
            self.loss_config      = config.loss
            self.type             = config.data.type

            self.mode             = "train"
            self.architecture     = list(self.network_config.architecture)
            self.epochs           = self.network_config.epochs
            self.learning_rate    = self.network_config.learning_rate
            self.decay_steps      = self.network_config.decay_steps
            self.warmup_steps     = self.network_config.warmup_steps
            self.save_filepath    = self.network_config.save_filepath
            self.load_filepath    = self.network_config.load_filepath

            self.seed             = config.system.seed
            self.train_batch_size = self.train_config.batch_size
            self.val_split        = self.train_config.val_split
            self.do_validation    = self.train_config.do_validation
            self.lambda_reg       = self.train_config.lambda_reg
            self.use_auto_lambda  = self.train_config.use_auto_lambda   

            # MLP ───────────────────────────────────────────────────────────
            if isinstance(config, MLPTrainConfig):
                self.physics_loss      = MLPLoss(self.loss_config)
                self.init_lambdas      = [self.loss_config.lambda_data, self.lambda_reg]
                self.physics_values, self.physics_learnable = {}, {}

            # PINN ───────────────────────────────────────────────────────────
            elif isinstance(config, PINNTrainConfig):
                self.physics_loss      = self.loss_config.residual_fn(self.loss_config)
                self.init_lambdas      = [self.loss_config.lambda_data, self.loss_config.lambda_physics, self.loss_config.lambda_boundary, self.lambda_reg,]
                self.physics_values, self.physics_learnable = self.extract_physics_params(self.loss_config.physics)

            # HNN ───────────────────────────────────────────────────────────
            elif isinstance(config, HNNTrainConfig):
                self.physics_loss      = HNNLoss(self.loss_config)
                self.init_lambdas      = [self.loss_config.lambda_eom, self.lambda_reg]
                self.physics_values, self.physics_learnable = self.extract_physics_params(self.loss_config.physics)

                if self.loss_config.hamiltonian.penalize_correction:
                    self.init_lambdas.insert(-1, self.loss_config.lambda_correction)

            # LNN ───────────────────────────────────────────────────────────
            elif isinstance(config, LNNTrainConfig):
                self.physics_loss      = LNNLoss(self.loss_config)
                self.init_lambdas      = [self.loss_config.lambda_eom, self.lambda_reg]
                self.physics_values, self.physics_learnable = self.extract_physics_params(self.loss_config.physics)

                if self.loss_config.lagrangian.penalize_correction:
                    self.init_lambdas.insert(-1, self.loss_config.lambda_correction)
        else:
            raise ValueError(f"Unknown config type: {type(config)}")
        

        self.rng          = random.PRNGKey(self.seed)
        self.rng_np       = self.seed
        self.architecture = [dict(layer) for layer in self.architecture]
        self.num_layers   = len(self.architecture)

        self.initiate_metrics()
        self.params               = []
        self.layer_configs        = []

        self.loss_names = (
            self.physics_loss.loss_names + ("reg",)
            if self.physics_loss is not None else ()
        )

        init_log_lambdas = jnp.array(
            [-0.5 * float(jnp.log(jnp.clip(jnp.array(2.0 * lam), 1e-8)))
            for lam in self.init_lambdas],
            dtype=jnp.float32,
        ) if self.init_lambdas else jnp.zeros(0, dtype=jnp.float32)

        all_params = {
            "net":         self.params,
            "log_lambdas": init_log_lambdas,
        }
        if self.physics_values:
            all_params["physics"] = self.physics_values

        if self.mode == "train":
            self.init_train(X, y)
        else:
            self.init_inference()

        self.layer_output_channels = [self.input_size[-1]]
        self.initializers()
        self.initialize_params()

        if self.mode == "train":
            total_steps       = self.epochs * max(1, self.N_train // self.train_batch_size)
            self.decay_steps  = self.decay_steps  if self.decay_steps  is not None else total_steps
            self.warmup_steps = self.warmup_steps if self.warmup_steps is not None else max(200, self.decay_steps // 40)

            self.lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.learning_rate,
                warmup_steps=self.warmup_steps,
                decay_steps=self.decay_steps,
                end_value=1e-5,
            )
            lambda_lr = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=1e-4,
                warmup_steps=self.warmup_steps,
                decay_steps=self.decay_steps,
                end_value=1e-5,
            )
            lambda_opt = optax.adam(lambda_lr) if self.use_auto_lambda else optax.set_to_zero()

            param_labels = {"net": "net", "log_lambdas": "lambda"}
            transforms = {
                "net": optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(self.lr_schedule, weight_decay=1e-5),
                ),
                "lambda": lambda_opt,
            }
            if self.physics_values:
                param_labels["physics"] = {
                    k: ("physics_learn" if self.physics_learnable.get(k, False) else "physics_fixed")
                    for k in self.physics_values
                }
                transforms["physics_learn"] = optax.adam(self.lr_schedule)
                transforms["physics_fixed"] = optax.set_to_zero()

            self.optimizer = optax.multi_transform(transforms, param_labels)
            self.state = TrainState(
                params=all_params,
                opt_state=self.optimizer.init(all_params),
                rng=self.rng,
            )

        else:
            self.optimizer = None
            self.state = TrainState(
                params=all_params,
                opt_state=None,
                rng=self.rng,
            )

        if self.load_filepath is not None:
            self.load_weights()
        self.warmup = False

    @staticmethod
    def extract_physics_params(physics_config):
        if physics_config is None:
            return {}, {}
        values    = {k: jnp.array(v.value) for k, v in physics_config.params.items()}
        learnable = {k: v.learnable for k, v in physics_config.params.items()}
        return values, learnable
    
    def init_train(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        X_train, X_val = self.make_train_and_val_points(X, X.shape[0], self.val_split)
        y_train, y_val = self.make_train_and_val_points(y, y.shape[0], self.val_split)

        self.N_train     = X_train.shape[0]
        self.input_size  = X_train.shape[1:]
        self.output_size = y_train.shape[1:]
        self.ndim        = len(self.input_size) - 1
        self.X_train_batched, self.y_train_batched = self.make_train_batch(self.train_batch_size, X_train, y_train)
        self.X_val_batched, self.y_val_batched     = self.make_train_batch(self.train_batch_size, X_val, y_val)

    def init_inference(self):
        return NotImplementedError

    def make_train_and_val_points(self, X, N, split=0.8):
        rng  = np.random.default_rng(self.rng_np)
        perm = rng.permutation(N)
        X    = X[perm]
        X_train = X[:int(N * split)]
        X_val   = X[int(N * split):]
        return X_train, X_val

    @staticmethod
    def make_train_batch(train_batch_size, X, y):
        N           = X.shape[0]
        num_batches = N // train_batch_size
        if num_batches == 0:
            raise ValueError("Batch size is larger than the number of train data points")
        X_batches = X[:num_batches * train_batch_size].reshape((num_batches, train_batch_size, *X.shape[1:]))
        y_batches = y[:num_batches * train_batch_size].reshape((num_batches, train_batch_size, *y.shape[1:]))
        return X_batches, y_batches

    def fc_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        rng, k      = random.split(rng)
        unit_size   = layer['units']
        activation  = layer['activation']
        if activation == "relu":
            W = random.normal(k, (unit_size, previous_channel_size)) * jnp.sqrt(2.0 / previous_channel_size)
        elif activation == "leaky_relu":
            slope = layer.get("slope", 0.01)
            W = random.normal(k, (unit_size, previous_channel_size)) * jnp.sqrt(2.0 / ((1 + slope**2) * previous_channel_size))
        elif activation in ["tanh", "sigmoid", "linear", "softmax", "sin", "gelu", "softplus"]:
            W = random.normal(k, (unit_size, previous_channel_size)) * jnp.sqrt(2.0 / (previous_channel_size + unit_size))
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        b = jnp.zeros((unit_size,))
        self.params.append(FCParams(W, b))
        self.layer_configs.append(FCLayerConfig(activation=activation, units=unit_size))
        previous_channel_size = unit_size
        return rng, previous_spatial_shape, previous_channel_size

    def flatten_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        self.params.append(None)
        self.layer_configs.append(FlattenConfig())
        previous_channel_size *= prod(previous_spatial_shape)
        previous_spatial_shape = ()
        return rng, previous_spatial_shape, previous_channel_size

    def layer_normalization_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        gamma = jnp.ones((previous_channel_size,))
        beta  = jnp.zeros((previous_channel_size,))
        self.params.append(NormParams(gamma, beta))
        self.layer_configs.append(LayerNormConfig())
        return rng, previous_spatial_shape, previous_channel_size

    def convolution_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        previous_height, previous_width = previous_spatial_shape
        rng, k1     = random.split(rng, 2)
        filters     = layer['filters']
        kernel_size = layer['kernel_size']
        stride      = layer.get('stride', 1)
        padding     = layer.get('padding', 0)
        activation  = layer['activation']
        strides     = (stride, stride)

        rhs_dilation = layer.get('rhs_dilation', (1, 1))
        lhs_dilation = layer.get('lhs_dilation', (1, 1))

        fan_in  = previous_channel_size * kernel_size * kernel_size
        fan_out = filters * kernel_size * kernel_size

        if activation == "relu":
            W = random.normal(k1, (kernel_size, kernel_size, previous_channel_size, filters)) * jnp.sqrt(2.0 / fan_in)
        elif activation == "leaky_relu":
            slope = layer.get("slope", 0.01)
            W = random.normal(k1, (kernel_size, kernel_size, previous_channel_size, filters)) * jnp.sqrt(2.0 / ((1 + slope**2) * fan_in))
        elif activation in ["tanh", "sigmoid", "linear", "softmax"]:
            W = random.normal(k1, (kernel_size, kernel_size, previous_channel_size, filters)) * jnp.sqrt(2.0 / (fan_in + fan_out))
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        b = jnp.zeros((filters,))
        self.params.append(ConvParams(W, b))

        padding_mode = "VALID" if padding == 0 else ((padding, padding), (padding, padding))
        self.layer_configs.append(ConvLayerConfig(
            strides=strides, padding=padding_mode, activation=activation,
            rhs_dilation=rhs_dilation, lhs_dilation=lhs_dilation,
            b_shape=(1,) * (ndim + 1) + (-1,),
        ))

        previous_height_eff = previous_height + (previous_height - 1) * (lhs_dilation[0] - 1)
        previous_width_eff  = previous_width  + (previous_width  - 1) * (rhs_dilation[1] - 1)
        kernel_eff_height   = kernel_size + (kernel_size - 1) * (rhs_dilation[0] - 1)
        kernel_eff_width    = kernel_size + (kernel_size - 1) * (rhs_dilation[1] - 1)

        previous_height       = (previous_height_eff - kernel_eff_height + 2 * padding) // stride + 1
        previous_width        = (previous_width_eff  - kernel_eff_width  + 2 * padding) // stride + 1
        previous_channel_size = filters
        return rng, (previous_height, previous_width), previous_channel_size

    def pool_max_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        kernel_size = layer['kernel_size']
        stride      = layer.get('stride', kernel_size)
        window      = (1, *(kernel_size for _ in range(ndim)), 1)
        strides     = (1, *(stride for _ in range(ndim)), 1)
        self.params.append(None)
        self.layer_configs.append(PoolMaxLayerConfig(window=window, strides=strides))
        previous_spatial_shape = tuple((d - kernel_size) // stride + 1 for d in previous_spatial_shape)
        return rng, previous_spatial_shape, previous_channel_size

    def pool_sum_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        kernel_size = layer['kernel_size']
        stride      = layer.get('stride', kernel_size)
        window      = (1, *(kernel_size for _ in range(ndim)), 1)
        strides     = (1, *(stride for _ in range(ndim)), 1)
        self.params.append(None)
        self.layer_configs.append(PoolSumLayerConfig(window=window, strides=strides))
        previous_spatial_shape = tuple((d - kernel_size) // stride + 1 for d in previous_spatial_shape)
        return rng, previous_spatial_shape, previous_channel_size

    def transformer_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        heads   = layer['heads']
        p_dropout = layer.get("p_dropout", 0.0)
        d_model = previous_channel_size
        d_head  = d_model // heads

        if d_model % heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads ({heads})")

        rng, k1, k2, k3, k4 = random.split(rng, 5)

        qk_scale = jnp.sqrt(1.0 / d_head)
        Wq = random.normal(k1, (d_model, d_model)) * qk_scale
        Wk = random.normal(k2, (d_model, d_model)) * qk_scale

        vo_scale = jnp.sqrt(1.0 / d_model)
        Wv = random.normal(k3, (d_model, d_model)) * vo_scale
        Wo = random.normal(k4, (d_model, d_model)) * vo_scale

        self.params.append(TransformerParams(Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo))
        self.layer_configs.append(TransformerConfig(heads=heads, d_head= d_head, scale=float(jnp.sqrt(d_head)), p_dropout=p_dropout))

        return rng, previous_spatial_shape, previous_channel_size
    
    def dropout_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        p = layer.get("p", 0.5)
        self.params.append(None)
        self.layer_configs.append(DropoutConfig(p=p))
        return rng, previous_spatial_shape, previous_channel_size

    def add_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        skip = layer["skip"]
        self.layer_configs.append(AddConfig(skip=skip))
        self.params.append(None)
        return rng, previous_spatial_shape, previous_channel_size

    def gpool_max_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        self.params.append(None)
        self.layer_configs.append(GPoolMaxConfig(ndim=ndim))
        previous_spatial_shape = ()
        return rng, previous_spatial_shape, previous_channel_size

    def gpool_sum_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        self.params.append(None)
        self.layer_configs.append(GPoolSumConfig(ndim=ndim))
        previous_spatial_shape = ()
        return rng, previous_spatial_shape, previous_channel_size

    def nearest_neighbour_upsampling_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        scaling = layer.get('scaling', 1)
        self.params.append(None)
        self.layer_configs.append(NearestNeighbourUpsamplingConfig(scaling=scaling, ndim=ndim))
        previous_spatial_shape = tuple(d * scaling for d in previous_spatial_shape)
        return rng, previous_spatial_shape, previous_channel_size

    def concatenation_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        skip = layer["skip"]
        self.layer_configs.append(ConcatenatingConfig(skip=skip))
        self.params.append(None)
        previous_channel_size = previous_channel_size + self.layer_output_channels[skip]
        return rng, previous_spatial_shape, previous_channel_size

    def bilinear_upsampling_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        scaling = layer.get('scaling', 1)
        self.params.append(None)
        previous_spatial_shape = tuple(d * scaling for d in previous_spatial_shape)
        self.layer_configs.append(BilinearUpsamplingConfig(scaling=scaling, ndim=ndim, new_spatial=previous_spatial_shape,))    
        return rng, previous_spatial_shape, previous_channel_size

    def bias_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        b = jnp.zeros((previous_channel_size,))
        self.params.append(BiasParams(b=b))
        self.layer_configs.append(BiasConfig(b_shape=(1,) * (ndim + 1) + (-1,),))
        return rng, previous_spatial_shape, previous_channel_size

    def flatten_spatial_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        self.params.append(None)
        n_tokens = prod(previous_spatial_shape)
        self.layer_configs.append(FlattenSpatialConfig(n_tokens=n_tokens))
        return rng, (n_tokens,), previous_channel_size

    def unflatten_spatial_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        previous_spatial_shape = layer["shape"]
        self.params.append(None)
        self.layer_configs.append(UnFlattenSpatialConfig(shape=previous_spatial_shape))
        return rng, previous_spatial_shape, previous_channel_size

    def positional_embedding_2d_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        height, width = previous_spatial_shape 
        d_model = previous_channel_size

        if d_model % 2 != 0:
            raise ValueError(f"d_model ({d_model}) must be even for factorised 2D positional embedding")

        rng, k1, k2 = random.split(rng, 3)
        scale = jnp.sqrt(1.0 / d_model)

        E_row = random.normal(k1, (height, d_model // 2)) * scale
        E_col = random.normal(k2, (width,  d_model // 2)) * scale

        self.params.append(PositionalEmbedding2DParams(E_row=E_row, E_col=E_col))
        self.layer_configs.append(PositionalEmbedding2DConfig(height=height, width=width))

        return rng, previous_spatial_shape, previous_channel_size

    def transformer_encoder_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        heads            = layer['heads']
        p_dropout_atten  = layer.get('p_dropout_atten', 0.0)
        p_dropout_fc     = layer.get('p_dropout_ffn', 0.0)
        d_model          = previous_channel_size
        d_head           = d_model // heads
        unit_size            = layer.get('unit_size', 4 * d_model)

        if d_model % heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads ({heads})")

        rng, k1, k2, k3, k4, k5, k6 = random.split(rng, 7)

        # ── Attention layer ─────────────────────────────────────────────────
        qk_scale = jnp.sqrt(1.0 / d_head)
        Wq = random.normal(k1, (d_model, d_model)) * qk_scale
        Wk = random.normal(k2, (d_model, d_model)) * qk_scale

        vo_scale = jnp.sqrt(1.0 / d_model)
        Wv = random.normal(k3, (d_model, d_model)) * vo_scale
        Wo = random.normal(k4, (d_model, d_model)) * vo_scale
        bo = jnp.zeros((d_model,))

        # ── FC layer ───────────────────────────────────────────────────────
        W1 = random.normal(k5, (unit_size, d_model)) * jnp.sqrt(2.0 / (d_model + unit_size))
        b1 = jnp.zeros((unit_size,))

        W2 = random.normal(k6, (d_model, unit_size)) * jnp.sqrt(2.0 / (d_model + unit_size))
        b2 = jnp.zeros((d_model,))

        # ── Layer norm ─────────────────────────────────
        gamma1 = jnp.ones( (d_model,))
        beta1  = jnp.zeros((d_model,))
        gamma2 = jnp.ones( (d_model,))
        beta2  = jnp.zeros((d_model,))

        self.params.append(TransformerEncoderParams(
            attn  = TransformerParams(Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo),
            bo    = bo,
            norm1 = NormParams(gamma=gamma1, beta=beta1),
            fc1  = FCParams(W=W1, b=b1),
            fc2  = FCParams(W=W2, b=b2),
            norm2 = NormParams(gamma=gamma2, beta=beta2),
        ))
        self.layer_configs.append(TransformerEncoderConfig(
            heads=heads,
            d_head=d_head,
            d_ff=unit_size,
            scale=float(jnp.sqrt(d_head)),
            p_dropout_atten=p_dropout_atten,
            p_dropout_fc=p_dropout_fc,
        ))

        return rng, previous_spatial_shape, previous_channel_size

    def branch_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        skip = layer["skip"]
        self.layer_configs.append(BranchConfig(skip=skip))
        self.params.append(None)
        previous_channel_size = self.layer_output_channels[skip]
        return rng, previous_spatial_shape, previous_channel_size

    def cross_attention_layer_initialization(self, rng, layer, ndim, previous_spatial_shape, previous_channel_size):
        heads      = layer["heads"]
        p_dropout  = layer.get("p_dropout", 0.0)
        d_model    = previous_channel_size
        d_head     = d_model // heads

        if d_model % heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads ({heads})")

        skip = layer["skip"]

        rng, k1, k2, k3, k4 = random.split(rng, 5)

        qk_scale = jnp.sqrt(1.0 / d_head)
        Wq = random.normal(k1, (d_model, d_model)) * qk_scale
        Wk = random.normal(k2, (d_model, d_model)) * qk_scale

        vo_scale = jnp.sqrt(1.0 / d_model)
        Wv = random.normal(k3, (d_model, d_model)) * vo_scale
        Wo = random.normal(k4, (d_model, d_model)) * vo_scale

        self.params.append(CrossAttentionParams(Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo))
        self.layer_configs.append(CrossAttentionConfig(heads=heads, d_head=d_head, scale=float(jnp.sqrt(d_head)), skip=skip, p_dropout=p_dropout,))

        return rng, previous_spatial_shape, previous_channel_size   

    def initiate_metrics(self):
        self.history = {"train": [], "val": [], "lambdas": []}

    def initializers(self):
        self.initializers = {
            "fc":                             self.fc_layer_initialization,
            "flatten":                        self.flatten_layer_initialization,
            "conv":                           self.convolution_layer_initialization,
            "layer_normalization":            self.layer_normalization_layer_initialization,
            "pool_max":                       self.pool_max_layer_initialization,
            "pool_sum":                       self.pool_sum_layer_initialization,
            "transformer":                    self.transformer_layer_initialization,
            "dropout":                        self.dropout_layer_initialization,
            "add":                            self.add_layer_initialization,
            "gpool_max":                      self.gpool_max_layer_initialization,
            "gpool_sum":                      self.gpool_sum_layer_initialization,
            "nearest_neighbour_upsampling":   self.nearest_neighbour_upsampling_layer_initialization,
            "concatenation":                  self.concatenation_layer_initialization,
            "bilinear_upsampling":            self.bilinear_upsampling_layer_initialization,
            "bias":                           self.bias_layer_initialization,
            "flatten_spatial":                self.flatten_spatial_layer_initialization,
            "unflatten_spatial":              self.unflatten_spatial_layer_initialization,
            "positional_embedding_2d":        self.positional_embedding_2d_layer_initialization,
            "transformer_encoder":            self.transformer_encoder_layer_initialization,
            "branch":                         self.branch_initialization,
            "cross_attention":                self.cross_attention_layer_initialization,  
        }

    def initialize_params(self):
        rng                    = self.rng
        previous_spatial_shape = self.input_size[:-1]
        previous_channel_size  = self.input_size[-1]
        for layer in self.architecture:
            layer_type  = layer['type']
            initializer = self.initializers.get(layer_type)
            if initializer is None:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            rng, previous_spatial_shape, previous_channel_size = initializer(
                rng, layer, self.ndim, previous_spatial_shape, previous_channel_size,
                )
            self.layer_output_channels.append(previous_channel_size)
        self.layer_configs_static = tuple(self.layer_configs)
        self.rng = rng

    activation_lookup = {
        "relu":       lambda x: jnp.maximum(0, x),
        "leaky_relu": lambda x: jnp.where(x > 0, x, 0.01 * x),
        "tanh":       jnp.tanh,
        "sigmoid":    jax.nn.sigmoid,
        "linear":     lambda x: x,
        "softmax":    lambda x: jax.nn.softmax(x, axis=-1),
        "softplus":   jax.nn.softplus,
        "gelu":       jax.nn.gelu,
        "sin":        jnp.sin,
    }

    @staticmethod
    def apply_activation(Z, activation):
        try:
            act_fn = NeuralNetwork.activation_lookup[activation]
        except KeyError:
            raise ValueError(f"Unsupported activation: {activation}")
        return act_fn(Z)

    @staticmethod
    def fc_layer_forward(layer_params, X, config):
        W = layer_params.W
        b = layer_params.b
        Z = jnp.dot(X, W.T) + b
        return NeuralNetwork.apply_activation(Z, config.activation)

    @staticmethod
    def flatten_layer_forward(layer_params, X, config):
        return X.reshape(X.shape[0], -1)

    @staticmethod
    def layer_normalization_layer_forward(layer_params, X, config):
        gamma, beta = layer_params
        mean   = jnp.mean(X, axis=-1, keepdims=True)
        var    = jnp.var(X,  axis=-1, keepdims=True)
        X_norm = (X - mean) / jnp.sqrt(var + 1e-5)
        return gamma * X_norm + beta

    @staticmethod
    def convolution_layer_forward(layer_params, X, config):
        W = layer_params.W
        b = layer_params.b
        Z = lax.conv_general_dilated(
            X, W,
            window_strides=config.strides,
            padding=config.padding,
            lhs_dilation=config.lhs_dilation,
            rhs_dilation=config.rhs_dilation,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        Z = Z + b.reshape(config.b_shape)
        return NeuralNetwork.apply_activation(Z, config.activation)

    @staticmethod
    def pool_max_layer_forward(layer_params, X, config):
        return lax.reduce_window(X, -jnp.inf, lax.max, config.window, config.strides, padding="VALID")

    @staticmethod
    def pool_sum_layer_forward(layer_params, X, config):
        return lax.reduce_window(X, 0.0, lax.add, config.window, config.strides, padding="VALID")

    @staticmethod
    def transformer_layer_forward(layer_params, X, config, rng, training):
        X = X[:, :, 0, :]

        B, T, d_model = X.shape

        Q = X @ layer_params.Wq.T
        K = X @ layer_params.Wk.T
        V = X @ layer_params.Wv.T

        Q = Q.reshape(B, T, config.heads, config.d_head).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, config.heads, config.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, config.heads, config.d_head).transpose(0, 2, 1, 3)

        scores  = (Q @ K.transpose(0, 1, 3, 2)) / config.scale
        weights = jax.nn.softmax(scores, axis=-1)

        weights = NeuralNetwork.dropout_layer_forward(
            None, weights, DropoutConfig(p=config.p_dropout), rng, training
        )

        out = weights @ V

        out = out.transpose(0, 2, 1, 3).reshape(B, T, d_model)
        out = out @ layer_params.Wo.T

        return out[:, :, None, :]

    @staticmethod
    def dropout_layer_forward(layer_params, X, config, rng, training=True):
        keep_prob = 1.0 - config.p
        def train_fn(X):
            mask = random.bernoulli(rng, keep_prob, X.shape)
            return (X * mask) / keep_prob
        def eval_fn(X):
            return X
        return lax.cond(training, train_fn, eval_fn, X)

    @staticmethod
    def add_layer_forward(layer_params, X, config, saves):
        return X + saves[config.skip]

    @staticmethod
    def gpool_max_layer_forward(layer_params, X, config):
        ndim = config.ndim
        axes = tuple(range(1, ndim + 1))
        return jnp.max(X, axis=axes)

    @staticmethod
    def gpool_sum_layer_forward(layer_params, X, config):
        ndim = config.ndim
        axes = tuple(range(1, ndim + 1))
        return jnp.sum(X, axis=axes)

    @staticmethod
    def nearest_neighbour_upsampling_layer_forward(layer_params, X, config):
        scaling = config.scaling
        ndim    = config.ndim
        for axis in range(1, ndim + 1):
            X = jnp.repeat(X, scaling, axis=axis)
        return X

    @staticmethod
    def concatenation_layer_forward(layer_params, X, config, saves):
        return jnp.concatenate([X, saves[config.skip]], axis=-1)

    @staticmethod
    def bilinear_upsampling_layer_forward(layer_params, X, config):
        B = X.shape[0]
        C = X.shape[-1]
        return jax.image.resize(X, (B, *config.new_spatial, C), method='linear')
    
    @staticmethod
    def bias_layer_forward(layer_params, X, config):
        return X + layer_params.b.reshape(config.b_shape)

    @staticmethod
    def flatten_spatial_layer_forward(layer_params, X, config):
        B = X.shape[0]
        C = X.shape[-1]
        return X.reshape(B, config.n_tokens, 1, C)
    
    @staticmethod
    def unflatten_spatial_layer_forward(layer_params, X, config):
        B = X.shape[0]
        C = X.shape[-1]
        return X.reshape(B, *config.shape, C)
    
    @staticmethod
    def positional_embedding_2d_layer_forward(layer_params, X, config):
        H = config.height
        W = config.width

        row_idx = jnp.repeat(jnp.arange(H), W)
        col_idx = jnp.tile(jnp.arange(W), H)

        row_emb = layer_params.E_row[row_idx]
        col_emb = layer_params.E_col[col_idx]

        pos_emb = jnp.concatenate([row_emb, col_emb], axis=-1)

        return X + pos_emb[None, :, None, :]

    @staticmethod
    def transformer_encoder_layer_forward(layer_params, X, config, rng, training):
        X = X[:, :, 0, :]
        B, T, d_model = X.shape
        residual = X

        # Normalization layer
        X_norm   = NeuralNetwork.layer_normalization_layer_forward(
            layer_params.norm1, X, LayerNormConfig()
        )
        attn_config = TransformerConfig(
            heads=config.heads, d_head=config.d_head,
            scale=config.scale, p_dropout=config.p_dropout_atten,
        )
        # Transformer layer
        out = NeuralNetwork.transformer_layer_forward(
            layer_params.attn, X_norm[:, :, None, :], attn_config, rng, training
        )
        out = out[:, :, 0, :] + layer_params.bo
        # Add layer
        X   = residual + out
        residual = X

        # Normalization layer
        X_norm   = NeuralNetwork.layer_normalization_layer_forward(
            layer_params.norm2, X, LayerNormConfig()
        )
        # FC layer
        Z = NeuralNetwork.fc_layer_forward(
            layer_params.fc1, X_norm, FCLayerConfig(activation="gelu", units=config.d_ff)
        )
        
        # Dropout layer
        Z = NeuralNetwork.dropout_layer_forward(
            None, Z, DropoutConfig(p=config.p_dropout_fc), rng, training
        )
        
        # FC layer
        Z = NeuralNetwork.fc_layer_forward(
            layer_params.fc2, Z, FCLayerConfig(activation="linear", units=d_model)
        )

        # Add layer
        X   = residual + Z

        return X[:, :, None, :]

    @staticmethod
    def branch_layer_forward(layer_params, X, config, saves):
        return saves[config.skip]

    @staticmethod
    def cross_attention_layer_forward(layer_params, X, config, rng, training, residuals):
        C = residuals[config.skip]
        C = C[:, :, 0, :]

        B, E, d_model = X.shape[0], X.shape[1], X.shape[3]
        X = X[:, :, 0, :]
        T = C.shape[1]

        Q = X @ layer_params.Wq.T 
        K = C @ layer_params.Wk.T  
        V = C @ layer_params.Wv.T 

        Q = Q.reshape(B, E, config.heads, config.d_head).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, config.heads, config.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, config.heads, config.d_head).transpose(0, 2, 1, 3)

        scores  = (Q @ K.transpose(0, 1, 3, 2)) / config.scale 
        weights = jax.nn.softmax(scores, axis=-1)

        weights = NeuralNetwork.dropout_layer_forward(
            None, weights, DropoutConfig(p=config.p_dropout), rng, training
        )

        out = weights @ V    
        out = out.transpose(0, 2, 1, 3).reshape(B, E, d_model)
        out = out @ layer_params.Wo.T

        return out[:, :, None, :]   

    layer_forward_int = (
        lambda p, X, c, rng, training, saves: NeuralNetwork.flatten_layer_forward(p, X, c),
        lambda p, X, c, rng, training, saves: NeuralNetwork.fc_layer_forward(p, X, c),
        lambda p, X, c, rng, training, saves: NeuralNetwork.convolution_layer_forward(p, X, c),
        lambda p, X, c, rng, training, saves: NeuralNetwork.pool_max_layer_forward(p, X, c),
        lambda p, X, c, rng, training, saves: NeuralNetwork.pool_sum_layer_forward(p, X, c),
        lambda p, X, c, rng, training, saves: NeuralNetwork.layer_normalization_layer_forward(p, X, c),
        lambda p, X, c, rng, training, saves: NeuralNetwork.transformer_layer_forward(p, X, c, rng, training),
        lambda p, X, c, rng, training, saves: NeuralNetwork.dropout_layer_forward(p, X, c, rng, training),
        lambda p, A, c, rng, training, saves: NeuralNetwork.add_layer_forward(p, A, c, saves),
        lambda p, A, c, rng, training, saves: NeuralNetwork.gpool_max_layer_forward(p, A, c),
        lambda p, A, c, rng, training, saves: NeuralNetwork.gpool_sum_layer_forward(p, A, c),
        lambda p, A, c, rng, training, saves: NeuralNetwork.nearest_neighbour_upsampling_layer_forward(p, A, c),
        lambda p, A, c, rng, training, saves: NeuralNetwork.concatenation_layer_forward(p, A, c, saves),
        lambda p, A, c, rng, training, saves: NeuralNetwork.bilinear_upsampling_layer_forward(p, A, c),
        lambda p, A, c, rng, training, saves: NeuralNetwork.bias_layer_forward(p, A, c),
        lambda p, A, c, rng, training, saves: NeuralNetwork.flatten_spatial_layer_forward(p, A, c),
        lambda p, A, c, rng, training, saves: NeuralNetwork.unflatten_spatial_layer_forward(p, A, c),
        lambda p, X, c, rng, training, saves: NeuralNetwork.positional_embedding_2d_layer_forward(p, X, c),
        lambda p, X, c, rng, training, saves: NeuralNetwork.transformer_encoder_layer_forward(p, X, c, rng, training),
        lambda p, X, c, rng, training, saves: NeuralNetwork.branch_layer_forward(p, X, c, saves),
        lambda p, X, c, rng, training, saves: NeuralNetwork.cross_attention_layer_forward(p, X, c, rng, training, saves),
    )

    @partial(jax.jit, static_argnames=("layer_configs_static", "layer_forward_int", "num_layers"))
    def forward_propagation(params, X, num_layers, layer_configs_static, layer_forward_int, rng, training=True):
        A       = X
        saves   = [A]
        configs = layer_configs_static
        forward = layer_forward_int
        for idx in range(num_layers):
            config      = configs[idx]
            layer_type  = config.type
            layer_params = params[idx]
            rng, subkey = random.split(rng)
            A = forward[layer_type](layer_params, A, config, subkey, training, saves)
            saves.append(A)
        return A

    @staticmethod
    def regularization_loss_function(params):
        leaves = jax.tree_util.tree_leaves(params)
        return sum(jnp.sum(leaf**2) for leaf in leaves if jnp.asarray(leaf).size > 0)

    @staticmethod
    def total_loss_function(params, X, y, aug_data, num_layers, layer_configs_static, physics_loss, layer_forward_int, rng, epoch=0, training=True):
        net_params  = params["net"]
        log_lambdas = params["log_lambdas"]
        log_lambdas = jnp.clip(log_lambdas, -5.0, 5.0)
        def forward_fn(net_params, X):
            return NeuralNetwork.forward_propagation(net_params, X, num_layers, layer_configs_static, layer_forward_int, rng, training)

        loss_r      = NeuralNetwork.regularization_loss_function(net_params)
        call_fn     = physics_loss if training else physics_loss.val_call
        all_losses  = call_fn(params, X, y, aug_data, forward_fn, epoch=epoch)
        all_losses = jnp.concatenate([all_losses, jnp.array([loss_r])])
        eff_lambdas = 0.5 * jnp.exp(-2.0 * log_lambdas)
        total_loss  = jnp.dot(eff_lambdas, all_losses) + jnp.sum(log_lambdas)
        return total_loss, all_losses

    @staticmethod
    @partial(jax.jit, static_argnames=("layer_configs_static", "physics_loss", "layer_forward_int", "num_layers", "optimizer"))
    def train_step_jitted(state, X, y, aug_data, num_layers, layer_configs_static, physics_loss, layer_forward_int, optimizer, epoch=0):
        params, opt_state, rng = state
        rng, subkey = random.split(rng)

        def loss_fn(p):
            total_loss, all_losses = NeuralNetwork.total_loss_function(
                p, X, y, aug_data, num_layers, layer_configs_static, physics_loss, layer_forward_int, subkey, epoch=epoch
            )
            return total_loss, all_losses

        (_, all_losses), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_state  = TrainState(new_params, new_opt_state, rng)
        return new_state, all_losses

    @staticmethod
    @partial(jax.jit, static_argnames=("layer_configs_static", "physics_loss", "layer_forward_int", "num_layers", "optimizer"))
    def train_epoch_jitted(state, X_train, y_train, aug_data, num_layers, layer_configs_static, physics_loss, layer_forward_int, optimizer, epoch=0):
        def train_step(state, batch):
            Xb, yb = batch
            new_state, losses = NeuralNetwork.train_step_jitted(
                state, Xb, yb, aug_data, num_layers, layer_configs_static, physics_loss, layer_forward_int, optimizer, epoch=epoch
            )
            return new_state, losses

        batches = (X_train, y_train)
        state, all_losses = lax.scan(train_step, state, batches)
        return state, all_losses

    @staticmethod
    @partial(jax.jit, static_argnames=("layer_configs_static", "physics_loss", "layer_forward_int", "num_layers"))
    def val_step_jitted(state, X_val, y_val, aug_data, num_layers, layer_configs_static, physics_loss, layer_forward_int):
        params = state.params
        rng    = random.PRNGKey(0)
        _, all_losses = NeuralNetwork.total_loss_function(
            params, X_val, y_val, aug_data, num_layers, layer_configs_static, physics_loss, layer_forward_int, rng, training=False,
        )
        return all_losses

    @staticmethod
    @partial(jax.jit, static_argnames=("layer_configs_static", "physics_loss", "layer_forward_int", "num_layers"))
    def val_epoch_jitted(state, X_val, y_val, aug_data, num_layers, layer_configs_static, physics_loss, layer_forward_int):
        def val_step(state, batch):
            Xb, yb = batch
            losses = NeuralNetwork.val_step_jitted(
                state, Xb, yb, aug_data, num_layers, layer_configs_static, physics_loss, layer_forward_int
            )
            return state, losses

        batches = (X_val, y_val)
        _, all_losses = lax.scan(val_step, state, batches)
        return all_losses

    @staticmethod
    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        else:
            return f"{s}s"

    @staticmethod
    def build_bar(progress, width=40):
        filled = int(progress * width)
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def print_training_ui(epoch, epochs, train_losses, val_losses, loss_names, log_lambdas, start_time, physics_desc=None):
        elapsed = time.time() - start_time
        it_s    = epoch / elapsed if elapsed > 0 else 0
        eta     = (epochs - epoch) / it_s if it_s > 0 else 0
        bar     = NeuralNetwork.build_bar(epoch / epochs)
        log_lambdas = np.clip(log_lambdas, -5.0, 5.0)
        eff_lam = 0.5 * np.exp(-2.0 * log_lambdas)

        t_mean = np.mean(train_losses, axis=0)
        v_mean = np.mean(val_losses, axis=0) if val_losses is not None else None

        total_train = float(np.dot(eff_lam, t_mean) + np.sum(log_lambdas))
        total_val   = float(np.dot(eff_lam, v_mean) + np.sum(log_lambdas)) if v_mean is not None else None

        lines = [f"Epoch {epoch}/{epochs}  {bar}"]
        v_str = f"  val={total_val:.3e}" if total_val is not None else ""
        lines.append(f"  {'total':<14}: train={total_train:.3e}{v_str}")
        for i, name in enumerate(loss_names):
            v_str = f"  val={v_mean[i]:.3e}" if v_mean is not None else ""
            lines.append(f"  {name:<14}: train={t_mean[i]:.3e}{v_str}  λ={eff_lam[i]:.3e}")
        if physics_desc:
            lines.append(f"  physics       : {physics_desc}")
        lines.append(
            f"  speed={it_s:.2f} it/s  "
            f"elapsed={NeuralNetwork.format_time(elapsed)}  "
            f"ETA={NeuralNetwork.format_time(eta)}"
        )

        n = len(lines)
        sys.stdout.write(f"\033[{n}F\033[J")
        print("\n".join(lines))

    def warmup_model(self):
        aux_data = self.physics_loss.aux_data

        X_step = jnp.zeros((self.train_batch_size, *self.input_size),  dtype=jnp.float32)
        y_step = jnp.zeros((self.train_batch_size, *self.output_size), dtype=jnp.float32)

        _ = NeuralNetwork.forward_propagation(
            self.state.params["net"], X_step, self.num_layers,
            self.layer_configs_static, self.layer_forward_int, self.state.rng,
        )
        _ = NeuralNetwork.train_step_jitted(
            self.state, X_step, y_step, aux_data,
            self.num_layers, self.layer_configs_static,
            self.physics_loss, self.layer_forward_int, self.optimizer,
        )
        if self.do_validation:
            _ = NeuralNetwork.val_step_jitted(
                self.state, X_step, y_step, aux_data,
                self.num_layers, self.layer_configs_static,
                self.physics_loss, self.layer_forward_int,
            )

        X_epoch = X_step[None]
        y_epoch = y_step[None]

        _ = NeuralNetwork.train_epoch_jitted(
            self.state, X_epoch, y_epoch, aux_data,
            self.num_layers, self.layer_configs_static,
            self.physics_loss, self.layer_forward_int, self.optimizer,
        )
        if self.do_validation:
            _ = NeuralNetwork.val_epoch_jitted(
                self.state, X_epoch, y_epoch, aux_data,
                self.num_layers, self.layer_configs_static,
                self.physics_loss, self.layer_forward_int,
            )

        jax.block_until_ready(self.state.params)
        self.warmup = True

    def training_stats(self):
        n_params = sum(
            x.size for x in jax.tree_util.tree_leaves(self.state.params["net"])
            if hasattr(x, "size")
        )
        devices = jax.devices()
        print(f"\n{'='*60}")
        print(f"{'Training configurations':^60}")
        print(f"{'='*60}")
        print(f"  device        : {devices[0].platform}")
        print(f"  n_params      : {n_params:,}")
        print(f"  input_size    : {self.input_size}")
        print(f"  output_size   : {self.output_size}")
        print(f"  epochs        : {self.epochs}")
        print(f"  batch_size    : {self.train_batch_size}")
        print(f"  lr_peak       : {self.learning_rate:.2e}")
        print(f"  auto_lambda   : {self.use_auto_lambda}")
        print(f"  lambda_reg    : {self.lambda_reg:.2e}")
        print(f"  loss_names    : {self.loss_names}")
        print(f"  do_validation : {self.do_validation}")
        print(f"  init time     : {NeuralNetwork.format_time(time.time() - self.t0_init)}")
        print(f"{'='*60}")
        print("\n" * (len(self.loss_names) + 4))
    
    def train_setup(self):
        if not self.warmup:
            print("Pre-jitting architecture")
            self.warmup_model()
            jax.block_until_ready(self.state.params)
            print("Pre-jitting complete")
            self.warmup = True

        self.train_start_time = time.time()
        self.training_stats()

    def train_epoch(self, epoch):
        aux_data   = self.physics_loss.aux_data
        loss_names = self.loss_names

        self.state, train_losses = NeuralNetwork.train_epoch_jitted(
            self.state,
            self.X_train_batched, self.y_train_batched, aux_data,
            self.num_layers, self.layer_configs_static, self.physics_loss,
            self.layer_forward_int, self.optimizer, epoch=jnp.asarray(epoch),
        )

        val_losses = None
        if self.do_validation:
            val_losses = NeuralNetwork.val_epoch_jitted(
                self.state,
                self.X_val_batched, self.y_val_batched, aux_data,
                self.num_layers, self.layer_configs_static, self.physics_loss,
                self.layer_forward_int,

            )

        log_lambdas  = jax.device_get(self.state.params["log_lambdas"])
        physics_desc = self.physics_loss.describe_physics(self.state.params)
        self.history["train"].append(jax.device_get(train_losses))
        self.history["val"].append(jax.device_get(val_losses) if val_losses is not None else None)
        self.history["lambdas"].append(log_lambdas)

        NeuralNetwork.print_training_ui(
            epoch, self.epochs, train_losses, val_losses,
            loss_names, log_lambdas, self.train_start_time, physics_desc,
        )

        if epoch % 5000 == 0:
            self.save_weights(training=True, epoch=epoch)

    def train(self):
        self.train_setup()
        for epoch in range(1, self.epochs + 1):
            self.train_epoch(epoch)
        self.save_weights()

    def save_weights(self, training=False, epoch=None):
        base_path = self.save_filepath
        dir_name  = os.path.dirname(base_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if training and epoch is not None:
            root, ext = os.path.splitext(base_path)
            save_path = f"{root}_epoch_{epoch}{ext}"
        else:
            save_path = base_path
        save_params = self.state.params
        bytes_out   = flax.serialization.to_bytes(save_params)

        with open(save_path, "wb") as f:
            f.write(bytes_out)

        if not training:
            print(f"Saved weights to {save_path}")

    def load_weights(self, inference=False):
        if not os.path.exists(self.load_filepath):
            raise FileNotFoundError(f"Weights file not found: {self.load_filepath}")
        with open(self.load_filepath, "rb") as f:
            bytes_in = f.read()
        new_params = flax.serialization.from_bytes(self.state.params, bytes_in)
        if self.mode == "train":
            self.state = TrainState(
                params=new_params,
                opt_state=self.optimizer.init(new_params),
                rng=self.state.rng,
            )
        else:
            self.state = TrainState(
                params=new_params,
                opt_state=None,
                rng=self.rng,
            )
        if not inference:
            print(f"Loaded weights from {self.load_filepath}")

    def release_gpu(self):
        self.state  = None
        self.params = None
        gc.collect()
        jax.clear_caches()  

    def predict(self, X):
        return NeuralNetwork.forward_propagation(
            self.state.params["net"], X, self.num_layers,
            self.layer_configs_static, self.layer_forward_int, self.state.rng, training=False,
        )

    def trajectories(self, t, state0=None):
        def forward_fn(net_params, X):
            return NeuralNetwork.forward_propagation(
                net_params, X, self.num_layers,
                self.layer_configs_static, self.layer_forward_int, self.state.rng, training=False,
            )

        t = jnp.asarray(t, dtype=jnp.float32)

        if self.type == "trajectory":
            field = self.physics_loss.trajectory(self.state.params, forward_fn)
            return np.asarray(jax.vmap(field)(t))

        if state0 is None:
            raise ValueError()

        field  = self.physics_loss.dynamics(self.state.params, forward_fn)
        state0 = jnp.asarray(state0, dtype=jnp.float32)
        dts    = t[1:] - t[:-1]

        def rk4_step(state, dt):
            k1 = field(state)
            k2 = field(state + 0.5 * dt * k1)
            k3 = field(state + 0.5 * dt * k2)
            k4 = field(state + dt * k3)
            new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return new_state, new_state

        _, states = lax.scan(rk4_step, state0, dts)
        return np.asarray(jnp.concatenate([state0[None], states], axis=0))

    def plot_training_history(self, log_scale=False, save_path=None, show=True):
        if not self.history["train"]:
            print("No training history to plot.")
            return None

        loss_names  = self.loss_names
        train_stack = np.stack([np.mean(np.asarray(epoch), axis=0) for epoch in self.history["train"]])
        has_val     = self.history["val"][0] is not None
        if has_val:
            val_stack = np.stack([np.mean(np.asarray(epoch), axis=0) for epoch in self.history["val"]])
        epochs = np.arange(1, train_stack.shape[0] + 1)

        log_lambdas_stack = np.stack(self.history["lambdas"])
        eff_lambdas_stack = 0.5 * np.exp(-2.0 * log_lambdas_stack)
        total_train = np.sum(eff_lambdas_stack * train_stack, axis=1) + np.sum(log_lambdas_stack, axis=1)
        if has_val:
            total_val = np.sum(eff_lambdas_stack * val_stack, axis=1) + np.sum(log_lambdas_stack, axis=1)

        n_losses = len(loss_names)
        fig, axes = plt.subplots(n_losses + 1, 1, figsize=(8, 3 * (n_losses + 1)), squeeze=False)

        ax = axes[0, 0]
        ax.plot(epochs, total_train, label="train")
        if has_val:
            ax.plot(epochs, total_val, label="val")
        if log_scale:
            ax.set_yscale("log")
        ax.set_title("total")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.grid(alpha=0.3)

        for i, name in enumerate(loss_names):
            ax = axes[i + 1, 0]
            ax.plot(epochs, train_stack[:, i], label="train")
            if has_val:
                ax.plot(epochs, val_stack[:, i], label="val")
            if log_scale:
                ax.set_yscale("log")
            ax.set_title(name)
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_lambdas_history(self, save_path=None, show=True):
        if not self.history["lambdas"]:
            print("No lambda history to plot.")
            return None

        loss_names     = self.loss_names
        log_lambdas    = np.stack(self.history["lambdas"])
        eff_lambdas    = 0.5 * np.exp(-2.0 * log_lambdas)
        epochs         = np.arange(1, log_lambdas.shape[0] + 1)

        fig, ax = plt.subplots(figsize=(8, 4))
        for i, name in enumerate(loss_names):
            ax.plot(epochs, eff_lambdas[:, i], label=name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("effective λ")
        ax.set_title("Kendall loss weights")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def benchmark(self, n_rounds=10, n_warmups=1):
        raise NotImplementedError()

# To implement: 
# Add Neural ODE layer
# Add Fourier Neural operator layer
# Add equivariant layer