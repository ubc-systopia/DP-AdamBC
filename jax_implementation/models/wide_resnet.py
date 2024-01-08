# Modified from https://github.com/deepmind/jax_privacy/blob/main/jax_privacy/src/training/image_classification/models/cifar.py

"""Definition of the CIFAR Wide Residual Network."""
import functools
import haiku as hk
import haiku.initializers as hk_init
import jax.numpy as jnp
import configlib
import jax

from models.model_utils import tempered_sigmoid, elementwise, WSConv2D


class WideResNet(hk.Module):
    """A Module defining a Wide ResNet."""

    def __init__(
            self,
            conf: configlib.Config,
            dropout_rate: float = 0.0,
            use_skip_init: bool = False,
            use_skip_paths: bool = True,
            groups: int = 16,
    ):
        super().__init__()
        self.conf = conf
        self.num_classes = conf.num_classes
        self.width = conf.resnet_width
        self.depth = conf.resnet_depth
        self.norm_fn = functools.partial(hk.GroupNorm, groups=groups)
        self.conv_fn = functools.partial(WSConv2D, w_init=hk_init.VarianceScaling(1.0))
        self.use_skip_init = use_skip_init
        self.use_skip_paths = use_skip_paths
        self.dropout_rate = dropout_rate
        self.resnet_blocks = (self.depth - 4) // 6

        if conf.activation == "relu":
            self.activation = jax.nn.relu
        elif conf.activation == "leaky_relu":
            self.activation = functools.partial(
                jax.nn.leaky_relu, negative_slope=self.conf.negative_slope
            )
        elif conf.activation == "tanh":
            self.activation = jnp.tanh
        elif conf.activation == "elu":
            self.activation = functools.partial(
                jax.nn.elu, alpha=self.conf.elu_alpha
            )
        elif conf.activation == "tempered_sigmoid":
            self.activation = elementwise(tempered_sigmoid, scale=1.58, inverse_temp=3.0, offset=0.71)
        else:
            raise NotImplementedError("Unknown activation function choice.")

    @hk.transparent
    def apply_skip_init(self, net, name):
        scale = hk.get_parameter(name, [1], init=jnp.zeros)
        return net * scale

    @hk.transparent
    def residual_block(self, net, width, strides, name):
        """Creates a residual block."""
        for i in range(self.resnet_blocks):
            if self.use_skip_paths:
                # This is the 'skip' branch.
                skip = net
                if i == 0:
                    skip = self.activation(skip)
                    skip = self.norm_fn(name=name + '_skip_norm')(skip)
                    skip = self.conv_fn(
                        width,
                        name=name + '_skip_conv',
                        stride=strides,
                        kernel_shape=(1, 1),
                    )(skip)
            # This is the 'residual' branch.
            for j in range(2):
                name_suffix = str(i) + '_' + str(j)
                strides = strides if name_suffix == '0_0' else (1, 1)
                net = self.activation(net)
                net = self.norm_fn(name=name + '_norm_' + name_suffix)(net)
                net = self.conv_fn(
                    width,
                    name=name + 'Conv_' + name_suffix,
                    kernel_shape=(3, 3),
                    stride=strides,
                )(net)
            # Merge both branches.
            if self.use_skip_init:
                net = self.apply_skip_init(net, name=name + 'Scale_' + name_suffix)
            if self.use_skip_paths:
                net += skip
        return net

    def __call__(self, inputs, forward_only=False):
        net = self.conv_fn(16, name='First_conv', kernel_shape=(3, 3))(inputs)
        net = self.residual_block(
            net, width=16 * self.width, strides=(1, 1), name='Block_1')
        net = self.residual_block(
            net, width=32 * self.width, strides=(2, 2), name='Block_2')
        net = self.residual_block(
            net, width=64 * self.width, strides=(2, 2), name='Block_3')
        net = self.activation(net)

        net = self.norm_fn(name='Final_norm')(net)

        net = jnp.mean(net, axis=[1, 2], dtype=jnp.float32)

        if self.dropout_rate > 0.0:
            dropout_rate = self.dropout_rate if not forward_only else 0.0
            net = hk.dropout(hk.next_rng_key(), dropout_rate, net)

        return hk.Linear(
            10,  # FIXME: self.num_output_classes
            w_init=hk_init.VarianceScaling(1.0),
            name='Softmax',
        )(net)
