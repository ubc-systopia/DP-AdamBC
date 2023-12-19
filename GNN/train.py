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

"""Training pipeline for DP-GNN."""

import functools
import os
import gc
from typing import Callable, Dict, Optional, Tuple
import torch

from absl import logging
import chex
from clu import checkpoint
from clu import metric_writers
from clu import parameter_overview
from clu import periodic_actions
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import numpy as np
import optax
import wandb
import random
import math

import input_pipeline
import models
import normalizations
import optimizers
import privacy_accountants
import sys

_SUBGRAPH_PADDING_VALUE = -1

@jax.jit
def compute_logits(state,
                   graph):
  """Computes unnormalized logits."""
  return state.apply_fn(state.params, graph).nodes


@jax.jit
def compute_loss(logits,
                 labels):
  """Computes the mean softmax cross-entropy loss."""
  assert labels.shape == logits.shape, f'Got incompatible shapes: logits as {logits.shape}, labels as {labels.shape}'
  loss = optax.softmax_cross_entropy(logits=logits, labels=labels)
  loss = jnp.mean(loss)
  return loss

@jax.jit
def compute_loss_multilabel(logits,
                            labels):
  """Computes the mean softmax cross-entropy loss or sigmoid cross-entropy if multilabel."""
  assert labels.shape == logits.shape, f'Got incompatible shapes: logits as {logits.shape}, labels as {labels.shape}'
  loss = jnp.sum(optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels), axis=-1)
  loss = jnp.mean(loss)
  return loss


def get_subgraphs(graph,
                  pad_to):
  """Creates an array of padded subgraphs."""
  num_nodes = jax.tree_util.tree_leaves(graph.nodes)[0].shape[0]
  outgoing_edges = {u: [] for u in range(num_nodes)}
  for sender, receiver in zip(graph.senders, graph.receivers):
    if sender != receiver:
      outgoing_edges[sender].append(receiver)

  subgraphs = np.zeros((num_nodes, pad_to), dtype=np.int32)
  for node in outgoing_edges:
    subgraph_indices = [node] + outgoing_edges[node]
    subgraph_indices = np.asarray(subgraph_indices)
    subgraph_indices = subgraph_indices[np.sort(
        np.unique(subgraph_indices, return_index=True)[1])]
    subgraph_indices = subgraph_indices[:pad_to]
    subgraphs[node] = np.pad(
        subgraph_indices, (0, pad_to - len(subgraph_indices)),
        'constant',
        constant_values=_SUBGRAPH_PADDING_VALUE)

  return subgraphs


@functools.partial(
    jax.jit, static_argnames=['add_reverse_edges', 'adjacency_normalization'])
def make_subgraph_from_indices(
    graph,
    subgraph_indices,
    add_reverse_edges,
    adjacency_normalization):
  """Constructs the subgraph corresponding to the given indices."""
  # Extract valid node indices.
  valid_mask = (subgraph_indices != _SUBGRAPH_PADDING_VALUE)

  # Node features.
  subgraph_nodes = graph.nodes[subgraph_indices]
  subgraph_nodes = jnp.where(valid_mask[:, jnp.newaxis], subgraph_nodes, 0.)

  # Add a dummy padding node.
  padding_node = len(subgraph_indices)
  subgraph_nodes = jnp.concatenate(
      (subgraph_nodes, jnp.zeros((1, subgraph_nodes.shape[1]))))
  subgraph_indices = jnp.where(valid_mask, subgraph_indices, padding_node)

  # Remap indices to within the subgraph.
  subgraph_senders = jnp.zeros_like(subgraph_indices, dtype=jnp.int32)
  subgraph_receivers = jnp.arange(len(subgraph_indices))

  # Handle padding.
  subgraph_senders = jnp.where(valid_mask, subgraph_senders, padding_node)
  subgraph_receivers = jnp.where(valid_mask, subgraph_receivers, padding_node)

  # Add reverse edges, ignoring self-loops.
  # This is generally not necessary, because we are only interested in the
  # predictions at the root node.
  if add_reverse_edges:
    subgraph_senders, subgraph_receivers = (
        jnp.concatenate((subgraph_senders, subgraph_receivers[1:])),
        jnp.concatenate((subgraph_receivers, subgraph_senders[1:])))

  # Build subgraph.
  subgraph_edges_unnormalized = jnp.ones_like(subgraph_senders)
  subgraph = graph._replace(
      nodes=subgraph_nodes,
      edges=subgraph_edges_unnormalized,
      senders=subgraph_senders,
      receivers=subgraph_receivers,
      n_node=jnp.expand_dims(subgraph_nodes.shape[0], axis=0),
      n_edge=jnp.expand_dims(subgraph_senders.shape[0], axis=0))

  # Normalize edge weights.
  return normalizations.normalize_edges_with_mask(subgraph, valid_mask,
                                                  adjacency_normalization)


@jax.jit
def compute_updates(state, graph,
                    labels,
                    node_indices):
  """Computes gradients for a single batch."""
  def loss_fn(params, graph,
              labels, node_indices):
    curr_state = state.replace(params=params)
    logits = compute_logits(curr_state, graph)
    logits = logits[node_indices]
    labels = labels[node_indices]
    return compute_loss(logits, labels)
  grads = jax.grad(loss_fn)(state.params, graph, labels, node_indices)
  return jax.tree_map(lambda grad: grad / grad.shape[0], grads)


@jax.jit
def reshape_before_pmap(arr):
  """Reshapes an array to have leading dimensions == jax.local_device_count()."""
  return arr.reshape((jax.local_device_count(),
                      arr.shape[0] // jax.local_device_count(), *arr.shape[1:]))


@jax.jit
def reshape_after_pmap(arr):
  """Undoes reshape_before_pmap()."""
  return arr.reshape((arr.shape[0] * arr.shape[1], *arr.shape[2:]))


@jax.jit
def get_dp_summed_grads(per_example_gradients,
                        l2_norm_threshold):
  """Requires per example gradients that are normalized by batch size. Standard clipping by L2 norm."""

  grad_norms = jax.tree_map(
      jax.vmap(jnp.linalg.norm),
      per_example_gradients)
  divisors = jax.tree_map(lambda g_norm: jnp.maximum(g_norm / l2_norm_threshold, 1.0), grad_norms)
  clipped_updates = jax.tree_map(jax.vmap(lambda g, div: g / div),
                                 per_example_gradients, divisors)
  return jax.tree_map(lambda g: jnp.sum(g, axis=0), clipped_updates)


@functools.partial(jax.jit, static_argnames=['adjacency_normalization', 'l2_norm_threshold', 'batch_size', 'multilabel'])
def compute_updates_for_dp(state,
                           graph, labels,
                           subgraphs,
                           adjacency_normalization,
                           l2_norm_threshold,
                           batch_size,
                           multilabel=False):
  """Computes gradients for a single batch for differentially private training."""

  def subgraph_loss(params, graph,
                    node_labels,
                    subgraph_indices):
    """Compute loss over this subgraph at the root node."""
    subgraph = make_subgraph_from_indices(
        graph,
        subgraph_indices,
        add_reverse_edges=False,
        adjacency_normalization=adjacency_normalization)
    subgraph_preds = state.apply_fn(params, subgraph).nodes
    node_preds = subgraph_preds[0, :]
    if multilabel:
      return compute_loss_multilabel(node_preds, node_labels)
    return compute_loss(node_preds, node_labels)
  # Get labels
  node_labels = labels
  # Get per example gradients
  per_example_gradient_fn = jax.vmap(jax.grad(subgraph_loss), in_axes=(None, None, 0, 0))
  per_example_grads = per_example_gradient_fn(state.params, graph, node_labels, subgraphs)
  # Batch-normalize
  normed_per_example_grads = jax.tree_map(lambda grad: grad / batch_size, per_example_grads)
  # Clip and sum
  summed_grads = get_dp_summed_grads(normed_per_example_grads, l2_norm_threshold)
  return summed_grads


@jax.jit
def update_model(state,
                 grads):
  """Applies updates to the parameters."""
  return state.apply_gradients(grads=grads)


@jax.jit
def evaluate_predictions(logits, labels,
                         mask):
  """Evaluates the model on the given dataset."""
  loss = optax.softmax_cross_entropy(logits, labels)
  loss = jnp.where(mask, loss, 0.)
  loss = jnp.sum(loss) / jnp.sum(mask)

  logits_match_labels = (jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
  logits_match_labels = jnp.where(mask, logits_match_labels, 0.)
  accuracy = jnp.sum(logits_match_labels) / jnp.sum(mask)
  return loss, accuracy


@jax.jit
def evaluate_predictions_multilabel(logits, labels,
                                    mask):
  """Evaluates the multilabel model on the given dataset."""
  loss = jnp.sum(optax.sigmoid_binary_cross_entropy(logits, labels), axis=-1)
  loss = jnp.where(mask, loss, 0.)
  loss = jnp.sum(loss) / jnp.sum(mask)
  preds = jnp.where((nn.sigmoid(logits) > 0.5), 1., 0.)

  TP = jnp.sum(jnp.logical_and(preds == 1, labels == 1))
  FP = jnp.sum(jnp.logical_and(preds == 1, labels == 0))
  FN = jnp.sum(jnp.logical_and(preds == 0, labels == 1))

  f1 = TP / (TP + 0.5*(FP + FN))
  return loss, f1


def create_model(config, graph,
                 rng):
  """Creates the model and initial parameters."""
  if config.model == 'mlp':
    model = models.GraphMultiLayerPerceptron(
        dimensions=([config.latent_size] * config.num_layers +
                    [config.num_classes]),
        activation=getattr(nn, config.activation_fn))
  elif config.model == 'gcn':
    model = models.GraphConvolutionalNetwork(
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_message_passing_steps=config.num_message_passing_steps,
        latent_size=config.latent_size,
        num_classes=config.num_classes,
        activation=getattr(nn, config.activation_fn))
  elif config.model == 'gat':
    model = models.GraphAttentionNetwork(
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_message_passing_steps=config.num_message_passing_steps,
        latent_size=config.latent_size,
        num_classes=config.num_classes,
        activation=getattr(nn, config.activation_fn),
        negative_slope=config.negative_slope,
        num_heads=config.num_heads,
        num_heads_decoder=config.num_heads_decoder,
        multilabel=config.multilabel)
  else:
    raise ValueError(f'Unsupported model: {config.model}.')
  print("initializing...")
  params = jax.jit(model.init)(rng, graph)
  print("initialized")
  return model, params


def compute_max_terms_per_node(config):
  """Returns the maximum number of gradient terms affected by a node.

  Args:
    config: The configuration dictionary.
      config.num_message_passing_steps must be one of 0, 1, or 2.
  """
  if config.model == 'mlp':
    return 1

  num_message_passing_steps = config.num_message_passing_steps
  max_node_degree = config.max_degree

  if num_message_passing_steps == 1:
    return max_node_degree + 1

  if num_message_passing_steps == 2:
    return max_node_degree ** 2 + max_node_degree + 1

  # We only support MLP and upto 2-layer GNNs.
  raise ValueError('Not supported for num_message_passing_steps > 2.')


def compute_base_sensitivity(config):
  """Returns the base sensitivity which is multiplied to the clipping threshold.

  Args:
    config: The configuration dictionary.
      config.num_message_passing_steps must be one of 0, 1, or 2.
  """
  if config.model == 'mlp':
    return 1.

  num_message_passing_steps = config.num_message_passing_steps
  max_node_degree = config.max_degree

  if num_message_passing_steps == 1:
    return float(2 * (max_node_degree + 1))

  if num_message_passing_steps == 2:
    return float(2 * (max_node_degree ** 2 + max_node_degree + 1))

  # We only support MLP and upto 2-layer GNNs.
  raise ValueError('Not supported for num_message_passing_steps > 2.')


def tree_flatten_1dim(tree):
    tree_flat, _ = jax.tree_util.tree_flatten(tree)
    return jnp.concatenate([i.flatten() for i in tree_flat])


def create_optimizer(
    apply_fn,
    params,
    config,
    graph,
    labels,
    subgraphs,
    rng,
    wandb_logging,
):
  """Creates the optimizer."""
  # TODO: single clipping threshold
  if config.differentially_private_training:
    base_sensitivity = compute_base_sensitivity(config)
    privacy_params = {
        'batch_size': config.batch_size,
        'l2_norm_threshold': config.l2_norm_threshold,
        'init_rng': rng,
        'base_sensitivity': base_sensitivity,
        'noise_multiplier': config.training_noise_multiplier,
    }
    
    sigma = optimizers.compute_opt_noise(config.l2_norm_threshold, base_sensitivity, 
                                          config.training_noise_multiplier)
    if wandb_logging:
      wandb.config.clipping_threshold = config.l2_norm_threshold
      wandb.config.sigma = sigma


  print("Got clipping thresholds")

  if config.optimizer == 'sgd':
    opt_params = {
        'learning_rate': config.learning_rate,
        'momentum': config.momentum,
        'nesterov': config.nesterov,
    }
    if config.differentially_private_training:
      return optimizers.dpsgd(**opt_params, **privacy_params)
    return optax.sgd(**opt_params)

  if config.optimizer == 'adam':
    opt_params = {
        'learning_rate': config.learning_rate,
        'b1': config.b1,
        'eps': config.eps
    }
    if config.differentially_private_training:
      return optimizers.dpadam(**opt_params, **privacy_params)
    return optax.adam(**opt_params)
  
  if config.optimizer == 'adamcorr':
    opt_params = {
        'learning_rate': config.learning_rate,
        'b1': config.b1,
        'eps_root': config.eps_root,
    }
    if config.differentially_private_training:
      return optimizers.dpadamcorr(**opt_params, **privacy_params)

  raise ValueError(f'Unsupported optimizer: {config.optimizer}')


def create_train_state(
    rng, config,
    graph,
    labels,
    subgraphs,
    wandb_logging,
):
  """Creates initial `TrainState`."""
  model_rng, opt_rng = jax.random.split(rng)
  model, params = create_model(config, graph, model_rng)
  apply_fn = jax.jit(model.apply)
  tx = create_optimizer(apply_fn, params, config, graph, labels, subgraphs, opt_rng, wandb_logging)
  return train_state.TrainState.create(
      apply_fn=apply_fn, params=params, tx=tx)


def get_max_training_epsilon(
    config):
  """Returns the privacy budget for DP training."""
  if not config.differentially_private_training:
    return None
  return config.max_training_epsilon


def compute_metrics(logits, labels,
                    masks, multilabel=False):
  if multilabel:
    train_loss, train_f1_micro = evaluate_predictions_multilabel(logits, labels,
                                                      masks['train'])
    val_loss, val_f1_micro = evaluate_predictions_multilabel(logits, labels,
                                                  masks['validation'])
    test_loss, test_f1_micro = evaluate_predictions_multilabel(logits, labels,
                                                    masks['test'])
    return {
      'train_loss': train_loss,
      'train_f1_micro': train_f1_micro,
      'val_loss': val_loss,
      'val_f1_micro': val_f1_micro,
      'test_loss': test_loss,
      'test_f1_micro': test_f1_micro,
    }
  else:
    train_loss, train_accuracy = evaluate_predictions(logits, labels,
                                                      masks['train'])
    val_loss, val_accuracy = evaluate_predictions(logits, labels,
                                                  masks['validation'])
    test_loss, test_accuracy = evaluate_predictions(logits, labels,
                                                    masks['test'])
    return {
      'train_loss': train_loss,
      'train_accuracy': train_accuracy,
      'val_loss': val_loss,
      'val_accuracy': val_accuracy,
      'test_loss': test_loss,
      'test_accuracy': test_accuracy,
    }


def log_metrics(step, metrics,
                adam_summary_stats,
                summary_writer, 
                wandb_logging,
                postfix = ''):
  """Logs all metrics."""
  # Formatting for accuracy.
  for metric in metrics:
    if 'accuracy' in metric:
      metrics[metric] *= 100

  # Add postfix to all metric names.
  metrics = {
      metric + postfix: metric_val for metric, metric_val in metrics.items()
  }

  # Log metrics to WandB
  if wandb_logging:
    wandb.log({**metrics, **adam_summary_stats})


def train_and_evaluate(config,
                       workdir):
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    The train state (which includes the `.params`).
  """
  # Clear GPU memory
  gc.collect()
  # Seed for reproducibility.
  rng = jax.random.PRNGKey(config.rng_seed)
  # Set the metric TODO: put in config
  eval_metric = 'test_f1_micro' if config.multilabel else 'test_accuracy'

  # Set up logging.
  wandb_logging = False
  if config.wandb_project is not None:
    wandb_logging = True
    wandb.init(project=config.wandb_project, config=dict(config), name=config.experiment_name, group=config.group)
  summary_writer = metric_writers.create_default_writer(workdir)
  summary_writer.write_hparams(dict(config))

  # Load base graph.
  rng, dataset_rng = jax.random.split(rng)
  base_graph = input_pipeline.load_graph(config)
  # Get datasets.
  dataset = input_pipeline.get_dataset(base_graph, config, dataset_rng)
  # graph, labels, masks = dataset
  graph, labels, masks, train_graph_index = jax.tree_map(jnp.asarray, dataset)
  # oh_labels = np.zeros((len(labels), config.num_classes))
  # oh_labels[np.arange(len(labels)), labels] = 1
  # labels = oh_labels
  if not config.multilabel:
    labels = jax.nn.one_hot(labels, config.num_classes)
  train_mask = masks['train']
  train_indices = jnp.where(train_mask)[0]
  # train_indices = np.where(train_mask)[0]
  train_labels = labels[train_indices]
  num_training_nodes = len(train_indices)

  gc.collect()

  # Get subgraphs.
  if config.differentially_private_training:
    print("Getting subgraphs")
    graph = jax.tree_map(np.asarray, graph)
    subgraphs = get_subgraphs(
        graph, pad_to=config.pad_subgraphs_to)
    graph = jax.tree_map(jnp.asarray, graph)

    # We only need the subgraphs for training nodes.
    train_subgraphs = subgraphs[train_indices]
    del subgraphs
  else:
    # TEMP ============================
    graph = jax.tree_map(np.asarray, graph)
    subgraphs = get_subgraphs(
        graph, pad_to=config.pad_subgraphs_to)
    graph = jax.tree_map(jnp.asarray, graph)

    # We only need the subgraphs for training nodes.
    train_subgraphs = subgraphs[train_indices]
    del subgraphs
    # TEMP ============================
    # train_subgraphs = None
  print('sampled subgraphs')

  # Initialize privacy accountant.
  print(num_training_nodes, compute_max_terms_per_node(config))
  training_privacy_accountant = privacy_accountants.get_training_privacy_accountant(
      config, num_training_nodes, compute_max_terms_per_node(config))

  print('created accountant')

  # Construct and initialize model.
  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng, config, graph, train_labels,
                             train_subgraphs, wandb_logging)

  print('initialized model (etc.)')

  # Set up checkpointing of the model.
  checkpoint_dir = os.path.join(workdir, 'checkpoints')
  ckpt = checkpoint.Checkpoint(checkpoint_dir, max_to_keep=2)
  state = ckpt.restore_or_initialize(state)
  # Get optimizer Adam summary stats
  if config.optimizer == 'sgd' or not config.differentially_private_training:
    adam_summary_stats = {}
  else:
    adam_summary_stats = state.opt_state[1][0].summary_stats
  initial_step = int(state.step) + 1

  # Log overview of parameters.
  parameter_overview.log_parameter_overview(state.params)

  # Log metrics after initialization.
  logits = compute_logits(state, graph)
  metrics_after_init = compute_metrics(logits, labels, masks, config.multilabel)
  metrics_after_init['epsilon'] = 0
  test_metric = metrics_after_init[eval_metric]
  log_metrics(
      0, metrics_after_init, adam_summary_stats, summary_writer, wandb_logging, postfix='')

  # Train model.
  rng, train_rng = jax.random.split(rng)
  max_training_epsilon = get_max_training_epsilon(config)

  # Hooks called periodically during training.
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_training_steps, writer=summary_writer)
  profiler = periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
  hooks = [report_progress, profiler]

  for step in range(initial_step, config.num_training_steps):
    # Perform one step of training.
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      # Sample batch.
      step_rng = jax.random.fold_in(train_rng, step)

      subgraphs, batch_labels = None, None
      if config.multi_graph:
        graph_indices = jax.random.choice(step_rng, train_graph_index.shape[0],
                                          (config.batch_size,))
        indices = train_graph_index[graph_indices, :].astype(int)
        for i in range(indices.shape[0]):
          if subgraphs is None:
            subgraphs = jnp.asarray(train_subgraphs[indices[0, 0]:indices[0, 1]])
          else:
            subgraphs = jnp.concatenate((subgraphs, train_subgraphs[indices[i, 0]:indices[i, 1]]))
          if batch_labels is None:
            batch_labels = train_labels[indices[0, 0]:indices[0, 1]]
          else:
            batch_labels = jnp.concatenate((batch_labels, train_labels[indices[i, 0]:indices[i, 1]]))
        del graph_indices, indices
      else:
        indices = jax.random.choice(step_rng, num_training_nodes,
                                    (config.batch_size,))
        subgraphs = jnp.asarray(train_subgraphs[indices])
        batch_labels = train_labels[indices]

      # Virtual batch sizes
      batch_size = subgraphs.shape[0]
      virtual_batch_num = math.ceil(batch_size / config.virtual_batch_size)
      grads = None
      for i in range(virtual_batch_num):
        start = i*config.virtual_batch_size
        end = (i+1)*config.virtual_batch_size
        subgraphs_indices = subgraphs[start:end]
        batch_label_indices = batch_labels[start:end]
        v_grads = compute_updates_for_dp(state, graph, batch_label_indices,
                                         subgraphs_indices, config.adjacency_normalization, 
                                         config.l2_norm_threshold, batch_size,
                                         config.multilabel)
        if grads is None:
          grads = v_grads
        else:
          grads = jax.tree_map(lambda grad, g: grad + g, grads, v_grads)
        del subgraphs_indices, batch_label_indices, v_grads
      del subgraphs, batch_labels

      # Update parameters.
      state = update_model(state, grads)
      # Get Adam optimizer summary stats
      if config.optimizer == 'sgd' or not config.differentially_private_training:
        adam_summary_stats = {}
      else:
        adam_summary_stats = state.opt_state[1][0].summary_stats
      
    # Quick indication that training is happening.
    logging.log_first_n(logging.INFO, 'Finished training step %d.', 10, step)
    for hook in hooks:
      hook(step)


    # Evaluate, if required.
    is_last_step = (step == config.num_training_steps - 1)
    if step % config.evaluate_every_steps == 0 or is_last_step:
      with report_progress.timed('eval'):
        # Check if privacy budget exhausted.
        training_epsilon = training_privacy_accountant(step + 1)
        if max_training_epsilon is not None and training_epsilon >= max_training_epsilon:
          break

        # Compute metrics.
        logits = compute_logits(state, graph)
        metrics_during_training = compute_metrics(logits, labels, masks, config.multilabel)
        metrics_during_training['epsilon'] = training_epsilon
        test_metric = metrics_during_training[eval_metric]
        log_metrics(
            step,
            metrics_during_training,
            adam_summary_stats, 
            summary_writer,
            wandb_logging)

    # Checkpoint, if required.
    if step % config.checkpoint_every_steps == 0 or is_last_step:
      print("checkpoint", int(step / config.checkpoint_every_steps), end="\r")
      with report_progress.timed('checkpoint'):
        ckpt.save(state)

    # Resample, if required.
    if config.resample_every_steps != 0 and (step+1) % config.resample_every_steps == 0:
      logging.info('resampling graph...')
      base_graph = input_pipeline.load_graph(config)
      old_num_training_nodes = num_training_nodes
      rng, dataset_rng = jax.random.split(rng)
      dataset = input_pipeline.get_dataset(base_graph, config, dataset_rng)
      graph, labels, masks, train_graph_index = jax.tree_map(jnp.asarray, dataset)
      labels = jax.nn.one_hot(labels, config.num_classes)
      train_mask = masks['train']
      train_indices = jnp.where(train_mask)[0]
      train_labels = labels[train_indices]
      num_training_nodes = len(train_indices)
      # Get subgraphs.
      if config.differentially_private_training:
        graph = jax.tree_map(np.asarray, graph)
        subgraphs = get_subgraphs(
            graph, pad_to=config.pad_subgraphs_to)
        graph = jax.tree_map(jnp.asarray, graph)

        # We only need the subgraphs for training nodes.
        train_subgraphs = subgraphs[train_indices]
        del subgraphs
      else:
        # TEMP ============================
        graph = jax.tree_map(np.asarray, graph)
        subgraphs = get_subgraphs(
            graph, pad_to=config.pad_subgraphs_to)
        graph = jax.tree_map(jnp.asarray, graph)

        # We only need the subgraphs for training nodes.
        train_subgraphs = subgraphs[train_indices]
        del subgraphs
      logging.info('resampling graph: node change %d -> %d, orginal graph has %d nodes', 
                   old_num_training_nodes, num_training_nodes, np.shape(base_graph.node_features)[0])
    # garbage collect
    gc.collect()

  print("done training")

  if wandb_logging:
    wandb.finish()
  gc.collect()

  return test_metric








# # Print the stuff for table
    # if step == 65 or step == 650:
    #   print(f"Step: {step}")
    #   cols = ['min', 'q25', 'median', 'q75', 'max', 'mean']
    #   rows = ['mt_clean_', 'mt_noised_', 'vt_clean_', 'vt_noised_', 'vt_corr_']
    #   print("Min, Q1, Median, Q3, Max, Mean")
    #   for row in rows:
    #     string = ""
    #     for col in cols:
    #       string += f"& {summary_stats[row + col]:.3e} "
    #     print(string)
    #   # deltas
    #   rows = ['vt_clean_', 'vt_noised_', 'vt_corr_']
    #   for row in rows:
    #     string = ""
    #     for col in cols:
    #       string += f"& {(summary_stats['mt_noised_' + col] / np.sqrt(summary_stats[row + col])):.3e} "
    #     print(string)