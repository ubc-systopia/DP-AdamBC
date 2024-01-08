# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines models."""

from typing import Callable, Sequence, Optional

import chex
from flax import linen as nn
import jax
import jax.numpy as jnp
import jraph


class MultiLayerPerceptron(nn.Module):
  """A multi-layer perceptron (MLP)."""

  latent_sizes: Sequence[int]
  activation: Optional[Callable[[chex.Array], chex.Array]]
  skip_connections: bool = False
  activate_final: bool = False

  @nn.compact
  def __call__(self, inputs):
    for index, dim in enumerate(self.latent_sizes):
      next_inputs = nn.Dense(dim)(inputs)

      if index != len(self.latent_sizes) - 1 or self.activate_final:
        if self.activation is not None:
          next_inputs = self.activation(next_inputs)

      if self.skip_connections and next_inputs.shape == inputs.shape:
        next_inputs = next_inputs + inputs

      inputs = next_inputs
    return inputs


class GraphMultiLayerPerceptron(nn.Module):
  """A multi-layer perceptron (MLP) applied to the node features."""

  dimensions: Sequence[int]
  activation: Callable[[chex.Array], chex.Array]

  @nn.compact
  def __call__(self, graph):
    mlp = MultiLayerPerceptron(
        self.dimensions,
        self.activation,
        skip_connections=False,
        activate_final=False)
    return graph._replace(nodes=mlp(graph.nodes))


class OneHopGraphAttention(nn.Module):
  """Performs one hop of a graph attention with unweighted edges."""

  update_fn: Callable[[chex.Array], chex.Array]
  negative_slope: float
  hop: int
  agg_function: str = 'concat' # either 'concat' or 'mean'
  num_heads: int = 1

  @nn.compact
  def __call__(self, graph):
    # Message-passing occurs against the direction of the input edges.
    senders, receivers = graph.receivers, graph.senders

    num_nodes = jax.tree_util.tree_leaves(graph.nodes)[0].shape[0]
    num_edges = senders.shape[0]

    # Multi-headed
    weighted_edges = None
    for k in range(self.num_heads):
      # Attention mechanism
      attention = MultiLayerPerceptron(
          [1],
          None,
          skip_connections=False,
          activate_final=False,
          name=f'attention_{self.hop}_{k}')

      # Attention stuff
      raw_attention = attention(jnp.concatenate((graph.nodes[senders], graph.nodes[receivers]), axis=1))
      nonlin_attention = nn.leaky_relu(raw_attention, negative_slope=self.negative_slope)
      attention_weights = jraph.segment_softmax(
          nonlin_attention,
          receivers,
          num_nodes,
          indices_are_sorted=False)
      # Same
      if weighted_edges is None:
        weighted_edges = attention_weights * graph.nodes[senders]
      else:
        if self.agg_function == 'concat':
          weighted_edges = jnp.concatenate((weighted_edges, attention_weights * graph.nodes[senders]), axis=1)
        else:
          weighted_edges = weighted_edges + attention_weights * graph.nodes[senders]
    # Complete the mean
    if self.agg_function == 'mean':
      weighted_edges = weighted_edges / self.num_heads
    # Concat across heads
    aggregated_nodes = jraph.segment_sum(
        weighted_edges,
        receivers,
        num_nodes,
        indices_are_sorted=False)

    # Update node features.
    aggregated_nodes = self.update_fn(aggregated_nodes)
    return graph._replace(nodes=aggregated_nodes)

class GraphAttentionNetwork(nn.Module):
  """A graph attention network from Velickovic, et al. (2018)."""

  latent_size: int
  num_classes: int
  num_message_passing_steps: int
  num_encoder_layers: int
  num_decoder_layers: int
  activation: Callable[[chex.Array], chex.Array]
  negative_slope: float
  num_heads: int
  multilabel: bool
  num_heads_decoder: int = 1

  @nn.compact
  def __call__(self, graph):
    # Encoder.
    encoder = MultiLayerPerceptron(
        [self.latent_size] * self.num_encoder_layers,
        self.activation,
        skip_connections=False,
        activate_final=True,
        name='encoder')
    graph = jraph.GraphMapFeatures(embed_node_fn=encoder)(graph)

    # Core.
    for hop in range(self.num_message_passing_steps):
      node_update_fn = MultiLayerPerceptron([self.latent_size],
                                            self.activation,
                                            skip_connections=True,
                                            activate_final=True,
                                            name=f'core_{hop}')
      core = OneHopGraphAttention(update_fn=node_update_fn, negative_slope=self.negative_slope, hop=hop, num_heads=self.num_heads)
      graph = core(graph)

    # Decoder.
    decoder_module = None
    if self.multilabel:
      # TODO: make this configurable
      decoder = MultiLayerPerceptron([self.num_classes],
                                     self.activation,
                                     skip_connections=False,
                                     activate_final=False,
                                     name='decoder')
      decoder_module = OneHopGraphAttention(update_fn=decoder, negative_slope=self.negative_slope, hop=self.num_message_passing_steps, num_heads=self.num_heads_decoder, agg_function='mean')
    else:
      decoder = MultiLayerPerceptron(
          [self.latent_size] * (self.num_decoder_layers - 1) + [self.num_classes],
          self.activation,
          skip_connections=False,
          activate_final=False,
          name='decoder')
      decoder_module = jraph.GraphMapFeatures(embed_node_fn=decoder)
    graph = decoder_module(graph)
    return graph

class OneHopGraphConvolution(nn.Module):
  """Performs one hop of a graph convolution with weighted edges."""

  update_fn: Callable[[chex.Array], chex.Array]
  num_partitions: int = 10

  @nn.compact
  def __call__(self, graph):
    # Message-passing occurs against the direction of the input edges.
    senders, receivers = graph.receivers, graph.senders

    num_nodes = jax.tree_util.tree_leaves(graph.nodes)[0].shape[0]
    num_edges = senders.shape[0]

    # Compute the convolution by partitioning the edges.
    # This saves a significant amount of memory.
    num_partitions = min(num_edges, self.num_partitions)
    partition_size = num_edges // num_partitions
    convolved_nodes = jnp.zeros_like(graph.nodes)
    for step in range(num_edges // partition_size + 1):
      partition_start = partition_size * step
      partition_end = partition_size * (step + 1)
      partition_end = min(partition_end, num_edges)
      partition_edges = graph.edges[partition_start:partition_end]
      partition_senders = senders[partition_start:partition_end]
      partition_receivers = receivers[partition_start: partition_end]
      weighted_edges = partition_edges * graph.nodes[partition_senders]
      convolved_nodes += jraph.segment_sum(
          weighted_edges,
          partition_receivers,
          num_nodes,
          indices_are_sorted=False)

    # Update node features.
    convolved_nodes = self.update_fn(convolved_nodes)
    return graph._replace(nodes=convolved_nodes)


class GraphConvolutionalNetwork(nn.Module):
  """A graph convolutional neural network from Kipf, et al. (2016)."""

  latent_size: int
  num_classes: int
  num_message_passing_steps: int
  num_encoder_layers: int
  num_decoder_layers: int
  activation: Callable[[chex.Array], chex.Array]

  @nn.compact
  def __call__(self, graph):
    # Encoder.
    encoder = MultiLayerPerceptron(
        [self.latent_size] * self.num_encoder_layers,
        self.activation,
        skip_connections=False,
        activate_final=True,
        name='encoder')
    graph = jraph.GraphMapFeatures(embed_node_fn=encoder)(graph)

    # Core.
    for hop in range(self.num_message_passing_steps):
      node_update_fn = MultiLayerPerceptron([self.latent_size],
                                            self.activation,
                                            skip_connections=True,
                                            activate_final=True,
                                            name=f'core_{hop}')
      core = OneHopGraphConvolution(update_fn=node_update_fn)
      graph = core(graph)

    # Decoder.
    decoder = MultiLayerPerceptron(
        [self.latent_size] * (self.num_decoder_layers - 1) + [self.num_classes],
        self.activation,
        skip_connections=False,
        activate_final=False,
        name='decoder')
    graph = jraph.GraphMapFeatures(embed_node_fn=decoder)(graph)
    return graph


# for step in range(num_edges // partition_size + 1):
#       partition_start = partition_size * step
#       partition_end = partition_size * (step + 1)
#       partition_end = min(partition_end, num_edges)
#       partition_senders = senders[partition_start:partition_end]
#       partition_receivers = receivers[partition_start: partition_end]
#       # Attention stuff
#       raw_attention = attention(jnp.concatenate((graph.nodes[partition_senders], graph.nodes[partition_receivers]), axis=1))
#       exp_nonlin_attention = jnp.exp(nn.leaky_relu(raw_attention, negative_slope=self.negative_slope))
#       attention_edges = exp_nonlin_attention * graph.nodes[partition_senders]
#       # print("receive:\n", partition_receivers)
#       # print("send:\n", partition_senders)
#       # print("alpha:\n", exp_nonlin_attention)
#       # Same
#       aggregated_nodes += jraph.segment_sum(
#           attention_edges,
#           partition_receivers,
#           num_nodes,
#           indices_are_sorted=False)
#       # Sum un-normed attention stuff for normalization
#       sum_exp += jraph.segment_sum(
#           exp_nonlin_attention,
#           partition_receivers,
#           num_nodes,
#           indices_are_sorted=False)