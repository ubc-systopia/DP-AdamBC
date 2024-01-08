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

"""Input pipeline for DP-GNN training."""

from typing import Dict, Tuple

import chex
import jraph
import ml_collections
import numpy as np
from scipy import sparse
import copy

import dataset_readers
import normalizations
import sampler


def add_reverse_edges(
    graph):
  """Add reverse edges to the graph."""
  senders = np.concatenate(
      (graph.senders, graph.receivers))
  receivers = np.concatenate(
      (graph.receivers, graph.senders))

  graph.senders = senders
  graph.receivers = receivers
  return graph

def filter_edge_index(senders, receivers, mask):
  # remap edges 
  # (i.e. if we remove 3, then edge 2 -> 4 becomes 2 -> 3)
  node_indices = np.arange(np.shape(mask)[0])[mask]
  edge_mapping = np.zeros(np.shape(mask)[0])
  edge_mapping[node_indices] = np.arange(np.shape(node_indices)[0])
  # remove edges containing nodes removed by mask
  edge_mask = np.logical_and(mask[senders], mask[receivers])
  return edge_mapping[senders[edge_mask]], edge_mapping[receivers[edge_mask]]


# make sparse adjacency matrix, A
def get_adjacency_matrix(senders, receivers, num_nodes):
    values = np.ones(np.shape(senders)[0])
    A = sparse.coo_array((values, (senders, receivers)), shape=(num_nodes, num_nodes))
    return A

  
def index_to_mask(n, index):
  mask = np.zeros(n, dtype=bool)
  mask[index] = True
  return mask

def mask_to_index(mask):
  return np.where(mask)


def sample_edgelists(graph, K, rng):
  rand_gen = np.random.default_rng(int(rng[0]))
  # graph = copy.copy(graph)
  x, y, senders, receivers = graph.node_features, graph.node_labels, graph.senders, graph.receivers
  n = np.shape(x)[0]
  print('num train_nodes', len(graph.train_nodes))
  train_mask = index_to_mask(n, graph.train_nodes)
  # only sample train edges
  train_edge_mask = train_mask[senders]
  # get out degrees
  A = get_adjacency_matrix(senders, receivers, n)
  eps = 1e-8
  out_degrees = A.sum(axis=1) + eps
  # sample out edges
  p = K / (2*out_degrees[senders])
  mask = rand_gen.random(np.shape(p)[0]) < p
  mask = np.logical_or(mask, ~train_edge_mask) # only do sampling on train edges!
  sampled_senders, sampled_receivers = senders[mask], receivers[mask]
  # check that no nodes have more in-degree than K
  A = get_adjacency_matrix(sampled_senders, sampled_receivers, n)
  out_degrees = A.sum(axis=1)
  node_mask = out_degrees <= K
  node_mask = np.logical_or(node_mask, ~train_mask) # only remove train nodes
  print('dropped count', len(np.where(node_mask == False)[0]))
  # filter out senders (edges) with out-degree greater than K
  mask = node_mask[sampled_senders]
  sampled_senders, sampled_receivers = sampled_senders[mask], sampled_receivers[mask]
  # save sampled edges
  graph.senders, graph.receivers = sampled_senders, sampled_receivers

  # NOTE: WE DON'T NEED TO SAMPLE THE NODES, JUST EDGES
  # graph.train_nodes = mask_to_index(train_mask[mask])
  # graph.validation_nodes = mask_to_index(index_to_mask(n, graph.validation_nodes)[mask])
  # graph.test_nodes = mask_to_index(index_to_mask(n, graph.test_nodes)[mask])
  # graph.node_features, graph.node_labels = x[mask], y[mask]
  # graph.senders, graph.receivers = filter_edge_index(sampled_senders, sampled_receivers, mask)

  return graph

def subsample_graph(graph, max_degree,
                    rng):
  """Subsamples the undirected input graph and returns a copy of the graph."""
  print("SAMPLING EDGELISTS")
  graph = sample_edgelists(graph, max_degree, rng)
  print("FINISHED!")

  # print("SAMPLING EDGELISTS")
  # edges = sampler.get_adjacency_lists(graph)
  # edges = sampler.sample_adjacency_lists(edges, graph.train_nodes, max_degree, rng)
  # senders = []
  # receivers = []
  # for u in edges:
  #   for v in edges[u]:
  #     senders.append(u)
  #     receivers.append(v)

  # graph.senders = senders
  # graph.receivers = receivers
  # print("FINISHED!")
  return graph


def compute_masks_for_splits(
    graph):
  """Compute boolean masks for the train, validation and test splits."""
  masks = {}
  num_nodes = graph.num_nodes()
  for split, split_nodes in zip(
      ['train', 'validation', 'test'],
      [graph.train_nodes, graph.validation_nodes, graph.test_nodes]):
    split_mask = np.zeros(num_nodes, dtype=bool)
    split_mask[split_nodes] = True
    masks[split] = split_mask
  return masks


def convert_to_graphstuple(
    graph):
  """Converts a dataset to one entire jraph.GraphsTuple, extracting labels."""
  return jraph.GraphsTuple(  # pytype: disable=wrong-arg-types  # jax-ndarray
      nodes=np.asarray(graph.node_features),
      edges=np.ones_like(graph.senders),
      senders=np.asarray(graph.senders),
      receivers=np.asarray(graph.receivers),
      globals=np.zeros(1),
      n_node=np.asarray([graph.num_nodes()]),
      n_edge=np.asarray([graph.num_edges()]),
  ), np.asarray(graph.node_labels)


def add_self_loops(graph):
  """Adds self-loops to the graph."""
  num_nodes = normalizations.compute_num_nodes(graph)
  senders = np.concatenate(
      (np.arange(num_nodes), np.asarray(graph.senders, dtype=np.int32)))
  receivers = np.concatenate(
      (np.arange(num_nodes), np.asarray(graph.receivers, dtype=np.int32)))

  return graph._replace(
      senders=senders,
      receivers=receivers,
      edges=np.ones_like(senders),
      n_edge=np.asarray([senders.shape[0]]))


def load_graph(config):
  """Load graph dataset."""
  graph = dataset_readers.get_dataset(config.dataset, config.dataset_path)
  graph = add_reverse_edges(graph)
  return graph

  
def get_dataset(graph, config, rng):
  """Sample graph and return graph dataset."""
  train_graph_index = None
  if config.multi_graph:
    train_graph_index = graph.train_graph_index
  graph = subsample_graph(graph, config.max_degree, rng)
  masks = compute_masks_for_splits(graph)
  graph, labels = convert_to_graphstuple(graph)
  graph = add_self_loops(graph)
  graph = normalizations.normalize_edges_with_mask(
      graph, mask=None, adjacency_normalization=config.adjacency_normalization)
  print("FINISHED DATASET LOADING")
  return graph, labels, masks, train_graph_index