import jax
import jax.numpy as jnp
import haiku as hk
import functools
import haiku.initializers as hk_init
import configlib

from models.model_utils import tempered_sigmoid, elementwise, WSConv2D


class CNN(hk.Module):
    def __init__(self, conf: configlib.Config):
        super().__init__()
        self.conf = conf
        self.num_classes = conf.num_classes

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

        if conf.normalization == "none":
            self.normalization = None
        elif conf.normalization == "group_norm":
            self.normalization = functools.partial(hk.GroupNorm, groups=16, create_scale=True, create_offset=True)
        else:
            raise NotImplementedError("Unknown normalization choice.")

        if conf.weight_standardization:
            self.conv_fn = functools.partial(WSConv2D, w_init=hk_init.VarianceScaling(1.0))
        else:
            self.conv_fn = hk.Conv2D


class CNN2(CNN):
    def __call__(self, x, forward_only=False):
        batch_size = x.shape[0]

        # Conv layer 1
        x = self.conv_fn(16, kernel_shape=8, stride=2, padding="VALID")(x)
        x = self.activation(x)
        x = hk.MaxPool(window_shape=2, strides=1, padding="VALID")(x)
        if self.normalization is not None:
            x = self.normalization(name='conv1_gn')(x)

        # Conv layer 2
        x = self.conv_fn(32, kernel_shape=4, stride=2)(x)
        x = self.activation(x)
        x = hk.MaxPool(window_shape=2, strides=1, padding="VALID")(x)
        if self.normalization is not None:
            x = self.normalization(name='conv2_gn')(x)

        x = x.reshape(batch_size, -1)

        # Fully connected layer 1
        x = hk.Linear(32)(x)
        x = self.activation(x)

        # Fully connected layer 2
        x = hk.Linear(self.num_classes)(x)

        return x


class CNN5(CNN):
    def __call__(self, x, forward_only=False):
        batch_size = x.shape[0]

        SAVE_ACT = False
        act_dict = {}

        # Conv layer 1
        x = self.conv_fn(32, kernel_shape=3, stride=1, padding="VALID")(x)
        x = self.activation(x)
        if SAVE_ACT:
            act_dict['l1'] = x.reshape(x.shape[0], -1)
        x = hk.MaxPool(window_shape=2, strides=2, padding="VALID")(x)
        if self.normalization is not None:
            x = self.normalization(name='conv1_gn')(x)

        # Conv layer 2
        x = self.conv_fn(64, kernel_shape=3, stride=1)(x)
        x = self.activation(x)
        if SAVE_ACT:
            act_dict['l2'] = x.reshape(x.shape[0], -1)
        x = hk.MaxPool(window_shape=2, strides=2, padding="VALID")(x)
        if self.normalization is not None:
            x = self.normalization(name='conv2_gn')(x)

        # Conv layer 3
        x = self.conv_fn(128, kernel_shape=3, stride=1)(x)
        x = self.activation(x)
        if SAVE_ACT:
            act_dict['l3'] = x.reshape(x.shape[0], -1)
        x = hk.MaxPool(window_shape=2, strides=2, padding="VALID")(x)
        if self.normalization is not None:
            x = self.normalization(name='conv3_gn')(x)

        # Conv layer 4
        x = self.conv_fn(256, kernel_shape=3, stride=1)(x)
        x = self.activation(x)
        if SAVE_ACT:
            act_dict['l4'] = x.reshape(x.shape[0], -1)
        if self.normalization is not None:
            x = self.normalization(name='conv4_gn')(x)

        # Conv layer 5
        x = self.conv_fn(self.num_classes, kernel_shape=3, stride=1)(x)

        x = hk.AvgPool(window_shape=x.shape[-2], strides=x.shape[-2], padding="VALID",
                       channel_axis=-1)(x)
        #  x = hk.MaxPool(window_shape=x.shape[-2], strides=1, padding="VALID")(x)

        x = x.reshape(batch_size, -1)

        return x
        # return x, act_dict
