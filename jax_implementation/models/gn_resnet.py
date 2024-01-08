# Adapted from
# https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/resnet.py
# ==============================================================================
"""Resnet."""

import types
from typing import Mapping, Optional, Sequence, Union, Any

import haiku as hk
from haiku._src import basic
from haiku._src import batch_norm
from haiku._src import conv
from haiku._src import module
from haiku._src import pool
import jax
import jax.numpy as jnp

import functools

import configlib

FloatStrOrBool = Union[str, float, bool]


class BlockV1(hk.Module):
  """ResNet V1 block with optional bottleneck."""

  def __init__(
      self,
      channels: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
      gn_config: Mapping[str, FloatStrOrBool],
      bottleneck: bool,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.use_projection = use_projection

    if self.use_projection:
      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding="SAME",
          name="shortcut_conv")

      self.proj_groupnorm = hk.GroupNorm(name="shortcut_groupnorm", **gn_config)

    channel_div = 4 if bottleneck else 1
    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=1 if bottleneck else 3,
        stride=1 if bottleneck else stride,
        with_bias=False,
        padding="SAME",
        name="conv_0")

    gn_0 = hk.GroupNorm(name="groupnorm_0", **gn_config)

    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=3,
        stride=stride if bottleneck else 1,
        with_bias=False,
        padding="SAME",
        name="conv_1")

    gn_1 = hk.GroupNorm(name="groupnorm_1", **gn_config)

    layers = ((conv_0, gn_0), (conv_1, gn_1))

    if bottleneck:
      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=1,
          with_bias=False,
          padding="SAME",
          name="conv_2")

      n_2 = hk.GroupNorm(name="groupnorm_2", scale_init=jnp.zeros, **gn_config)
      layers = layers + ((conv_2, gn_2),)

    self.layers = layers

  def __call__(self, inputs):
    out = shortcut = inputs

    if self.use_projection:
      shortcut = self.proj_conv(shortcut)
      shortcut = self.proj_groupnorm(shortcut)

    for i, (conv_i, gn_i) in enumerate(self.layers):
      out = conv_i(out)
      out = gn_i(out)
      if i < len(self.layers) - 1:  # Don't apply relu on last layer
        out = jax.nn.relu(out)

    return jax.nn.relu(out + shortcut)


class BlockV2(hk.Module):
  """ResNet V2 block with optional bottleneck."""

  def __init__(
      self,
      channels: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
      gn_config: Mapping[str, FloatStrOrBool],
      bottleneck: bool,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.use_projection = use_projection

    if self.use_projection:
      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding="SAME",
          name="shortcut_conv")

    channel_div = 4 if bottleneck else 1
    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=1 if bottleneck else 3,
        stride=1 if bottleneck else stride,
        with_bias=False,
        padding="SAME",
        name="conv_0")

    gn_0 = hk.GroupNorm(name="groupnorm_0", **gn_config)

    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=3,
        stride=stride if bottleneck else 1,
        with_bias=False,
        padding="SAME",
        name="conv_1")

    gn_1 = hk.GroupNorm(name="groupnorm_1", **gn_config)

    layers = ((conv_0, gn_0), (conv_1, gn_1))

    if bottleneck:
      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=1,
          with_bias=False,
          padding="SAME",
          name="conv_2")

      # NOTE: Some implementations of ResNet50 v2 suggest initializing
      # gamma/scale here to zeros.
      gn_2 = hk.GroupNorm(name="groupnorm_2", **gn_config)

      layers = layers + ((conv_2, gn_2),)

    self.layers = layers

  def __call__(self, inputs):
    x = shortcut = inputs

    for i, (conv_i, gn_i) in enumerate(self.layers):
      x = gn_i(x)
      x = jax.nn.relu(x)
      if i == 0 and self.use_projection:
        shortcut = self.proj_conv(x)
      x = conv_i(x)

    return x + shortcut


class BlockGroup(hk.Module):
  """Higher level block for ResNet implementation."""

  def __init__(
      self,
      channels: int,
      num_blocks: int,
      stride: Union[int, Sequence[int]],
      gn_config: Mapping[str, FloatStrOrBool],
      resnet_v2: bool,
      bottleneck: bool,
      use_projection: bool,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    block_cls = BlockV2 if resnet_v2 else BlockV1

    self.blocks = []
    for i in range(num_blocks):
      self.blocks.append(
          block_cls(channels=channels,
                    stride=(1 if i else stride),
                    use_projection=(i == 0 and use_projection),
                    bottleneck=bottleneck,
                    gn_config=gn_config,
                    name="block_%d" % (i)))

  def __call__(self, inputs):
    out = inputs
    for block in self.blocks:
      out = block(out)
    return out


def check_length(length, value, name):
  if len(value) != length:
    raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


class ResNet(hk.Module):
  """ResNet model."""

  CONFIGS = {
      18: {
          "blocks_per_group": (2, 2, 2, 2),
          "bottleneck": False,
          "channels_per_group": (64, 128, 256, 512),
          "use_projection": (False, True, True, True),
      },
      34: {
          "blocks_per_group": (3, 4, 6, 3),
          "bottleneck": False,
          "channels_per_group": (64, 128, 256, 512),
          "use_projection": (False, True, True, True),
      },
      50: {
          "blocks_per_group": (3, 4, 6, 3),
          "bottleneck": True,
          "channels_per_group": (256, 512, 1024, 2048),
          "use_projection": (True, True, True, True),
      },
  }

  BlockGroup = BlockGroup  # pylint: disable=invalid-name
  BlockV1 = BlockV1  # pylint: disable=invalid-name
  BlockV2 = BlockV2  # pylint: disable=invalid-name

  def __init__(
      self,
      blocks_per_group: Sequence[int],
      num_classes: int,
      gn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      resnet_v2: bool = False,
      bottleneck: bool = True,
      channels_per_group: Sequence[int] = (256, 512, 1024, 2048),
      use_projection: Sequence[bool] = (True, True, True, True),
      logits_config: Optional[Mapping[str, Any]] = None,
      name: Optional[str] = None,
      initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      strides: Sequence[int] = (1, 2, 2, 2),
  ):
    """Constructs a ResNet model.
    Args:
      blocks_per_group: A sequence of length 4 that indicates the number of
        blocks created in each group.
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers. By default the
        ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        ``False``.
      bottleneck: Whether the block should bottleneck or not. Defaults to
        ``True``.
      channels_per_group: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
      use_projection: A sequence of length 4 that indicates whether each
        residual block should use projection.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
      strides: A sequence of length 4 that indicates the size of stride
        of convolutions for each block in each group.
    """
    super().__init__(name=name)
    self.resnet_v2 = resnet_v2

    gn_config = dict(gn_config or {})
    gn_config.setdefault("groups", 32)
    gn_config.setdefault("eps", 1e-5)
    gn_config.setdefault("create_scale", True)
    gn_config.setdefault("create_offset", True)

    logits_config = dict(logits_config or {})
    logits_config.setdefault("w_init", jnp.zeros)
    logits_config.setdefault("name", "logits")

    # Number of blocks in each group for ResNet.
    check_length(4, blocks_per_group, "blocks_per_group")
    check_length(4, channels_per_group, "channels_per_group")
    check_length(4, strides, "strides")

    initial_conv_config = dict(initial_conv_config or {})
    initial_conv_config.setdefault("output_channels", 64)
    initial_conv_config.setdefault("kernel_shape", 7)
    initial_conv_config.setdefault("stride", 2)
    initial_conv_config.setdefault("with_bias", False)
    initial_conv_config.setdefault("padding", "SAME")
    initial_conv_config.setdefault("name", "initial_conv")

    self.initial_conv = hk.Conv2D(**initial_conv_config)

    if not self.resnet_v2:
        self.initial_groupnorm = hk.GroupNorm(name="initial_groupnorm", **gn_config)

    self.block_groups = []
    for i, stride in enumerate(strides):
      self.block_groups.append(
          BlockGroup(channels=channels_per_group[i],
                     num_blocks=blocks_per_group[i],
                     stride=stride,
                     gn_config=gn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=bottleneck,
                     use_projection=use_projection[i],
                     name="block_group_%d" % (i)))

    if self.resnet_v2:
        self.final_groupnorm = hk.GroupNorm(name="final_groupnorm", **gn_config)

    self.logits = hk.Linear(num_classes, **logits_config)

  def __call__(self, inputs):
    out = inputs
    out = self.initial_conv(out)
    if not self.resnet_v2:
      out = self.initial_groupnorm(out)
      out = jax.nn.relu(out)

    out = hk.max_pool(out,
                      window_shape=(1, 3, 3, 1),
                      strides=(1, 2, 2, 1),
                      padding="SAME")

    for block_group in self.block_groups:
      out = block_group(out)

    if self.resnet_v2:
      out = self.final_groupnorm(out)
      out = jax.nn.relu(out)
    out = jnp.mean(out, axis=(1, 2))
    return self.logits(out)


class GNResNet18(ResNet):
  """ResNet18."""

  def __init__(self, conf: configlib.Config,):
    """Constructs a ResNet model.
    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
      strides: A sequence of length 4 that indicates the size of stride
        of convolutions for each block in each group.
    """

    # This is for CIFAR10 and other low-res datasets. TODO: Adapt to datasets.
    initial_conv_config = {
        "output_channels": 64,
        "kernel_shape": 3,
        "stride": 1,
        "with_bias": True,
        "padding": "SAME",
        "name": "initial_conv"}

    super().__init__(num_classes=conf.num_classes,
                     initial_conv_config=initial_conv_config,
                     **ResNet.CONFIGS[18])
    self.conf = conf


class ResNet34(ResNet):
  """ResNet34."""

  def __init__(
      self,
      num_classes: int,
      bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      resnet_v2: bool = False,
      logits_config: Optional[Mapping[str, Any]] = None,
      name: Optional[str] = None,
      initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      strides: Sequence[int] = (1, 2, 2, 2),
  ):
    """Constructs a ResNet model.
    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
      strides: A sequence of length 4 that indicates the size of stride
        of convolutions for each block in each group.
    """
    super().__init__(num_classes=num_classes,
                     bn_config=bn_config,
                     initial_conv_config=initial_conv_config,
                     resnet_v2=resnet_v2,
                     strides=strides,
                     logits_config=logits_config,
                     name=name,
                     **ResNet.CONFIGS[34])


class ResNet50(ResNet):
  """ResNet50."""

  def __init__(
      self,
      num_classes: int,
      bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      resnet_v2: bool = False,
      logits_config: Optional[Mapping[str, Any]] = None,
      name: Optional[str] = None,
      initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      strides: Sequence[int] = (1, 2, 2, 2),
  ):
    """Constructs a ResNet model.
    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
      strides: A sequence of length 4 that indicates the size of stride
        of convolutions for each block in each group.
    """
    super().__init__(num_classes=num_classes,
                     bn_config=bn_config,
                     initial_conv_config=initial_conv_config,
                     resnet_v2=resnet_v2,
                     strides=strides,
                     logits_config=logits_config,
                     name=name,
                     **ResNet.CONFIGS[50])

