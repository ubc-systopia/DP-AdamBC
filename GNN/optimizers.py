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

"""Optimizers."""

import chex
import jax
import jax.numpy as jnp
import optax
from optim import adam, adamcorr

def compute_opt_noise(l2_norm_threshold, base_sensitivity, noise_multiplier):
  return l2_norm_threshold * base_sensitivity * noise_multiplier


def dp_aggregate(
    batch_size,
    l2_norm_threshold,
    base_sensitivity,
    noise_multiplier,
    init_rng,
    return_type='original',
):
  """Aggregates gradients based on the DP-SGD algorithm.

  This method adds noise to the summed clipped gradients (that have been normalized by batch size).

  WARNING: Unlike other transforms, `dp_aggregate` expects
  the input updates to have a batch dimension in the 0th axis. That is, this
  function expects per-example gradients as input (which are easy to obtain in
  JAX using `jax.vmap`). It can still be composed with other transformations as
  long as it is the first in the chain.
  Further, each per-example gradient must already be divided by the batch size.

  References:
    [Abadi et al, 2016](https://arxiv.org/abs/1607.00133)

  Args:
    batch_size: size of each batch
    l2_norm_threshold: max L2 norm of the per-example gradients across all layers.
    base_sensitivity: ratio of sensitivity to the clipping norm.
    noise_multiplier: ratio of noise standard deviation to the sensitivity.
    return_type: 'original' or 'custom', determines if summed updates should be included too ('custom')
    init_rng: initial jax.random.PRNGKey

  Returns:
    A `GradientTransformation`.
  """
  noise_std = compute_opt_noise(l2_norm_threshold, base_sensitivity, noise_multiplier)

  def init_fn(params):
    del params
    return optax.DifferentiallyPrivateAggregateState(
        rng_key=init_rng)

  def update_fn(summed_updates, state, params):
    del params
    grads_flat, grads_treedef = jax.tree_util.tree_flatten(summed_updates)

    new_key, *rngs = jax.random.split(state.rng_key, len(grads_flat) + 1)
    rng_tree = jax.tree_util.tree_unflatten(grads_treedef, rngs)

    noise = jax.tree_map(
        lambda g, rng: (noise_std * jax.random.normal(rng, g.shape, g.dtype)),
        summed_updates, rng_tree)
    noisy_updates = jax.tree_map(lambda g, noise: (g + noise), summed_updates,
                                 noise)
    if return_type == 'original':
      return (noisy_updates,
              optax.DifferentiallyPrivateAggregateState(rng_key=new_key))
    else:
      return ((summed_updates, noisy_updates),
              optax.DifferentiallyPrivateAggregateState(rng_key=new_key))

  return optax.GradientTransformation(init_fn, update_fn)


def dpsgd(batch_size, learning_rate, l2_norm_threshold,
          base_sensitivity, noise_multiplier,
          init_rng, momentum,
          nesterov):
  """A differentially-private version of SGD."""
  return optax.chain(
      dp_aggregate(batch_size, l2_norm_threshold, base_sensitivity, noise_multiplier,
                   init_rng), optax.sgd(learning_rate, momentum, nesterov))


# def dpadam(learning_rate, l2_norms_threshold,
#            base_sensitivity, noise_multiplier,
#            init_rng):
#   """A differentially-private version of Adam."""
#   return optax.chain(
#       dp_aggregate(l2_norms_threshold, base_sensitivity, noise_multiplier,
#                    init_rng), optax.adam(learning_rate))

def dpadam(batch_size, learning_rate, b1, eps, l2_norm_threshold,
           base_sensitivity, noise_multiplier,
           init_rng):
  """A differentially-private version of Adam."""
  b2 = 1 - (1 - b1)**2
  return optax.chain(
      dp_aggregate(batch_size, l2_norm_threshold, base_sensitivity, noise_multiplier,
                   init_rng, return_type='custom'), adam(learning_rate, b1, b2, eps))


def dpadamcorr(batch_size, learning_rate, b1, eps_root, l2_norm_threshold,
               base_sensitivity, noise_multiplier, init_rng):
  """A differentially-private version of Adam Corr."""
  b2 = 1 - (1 - b1)**2
  sigma = compute_opt_noise(l2_norm_threshold, base_sensitivity, noise_multiplier)
  return optax.chain(
      dp_aggregate(batch_size, l2_norm_threshold, base_sensitivity, noise_multiplier, init_rng,
                   return_type='custom'), adamcorr(sigma, learning_rate, b1, b2, 0, eps_root))
