import numpy as np

from trainer.iterative import IterativeTrainer
from trainer.utils import grad_norm, multiply_along_axis, _calculate_angle
from data_utils.augmult import apply_augmult

import jax
import jax.numpy as jnp
from opacus.data_loader import DPDataLoader
from opacus.accountants.rdp import RDPAccountant
from opacus.accountants.prv import PRVAccountant
from opacus.accountants.utils import get_noise_multiplier
import math

import configlib
import time
from functools import partial
import pickle
import os

parser = configlib.add_parser("DP Trainer config")
parser.add_argument("--dp_sampling", default=False, action='store_true',
                    help="Use proper DP sampling for amplification with https//opacus.ai/api/data_loader.html.")
parser.add_argument("--dp_noise_multiplier", default=1., type=float,
                    help="The noise multiple for DP-SGD and derivatives.")
parser.add_argument("--dp_l2_norm_clip", default=1., type=float,
                    help="The L2 clipping value for per example gradients for DP-SGD and derivatives.")
parser.add_argument("--virtual_batch_size", "--vb", type=int, metavar="VBACH_SIZE",
                    help="Use virual baches of size VBACH_SIZE iff batch_size > VBACH_SIZE.")
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--augmult_num', type=int, default=0)
parser.add_argument('--priv_accountant', choices=['rdp', 'prv'], default='prv')
parser.add_argument('--target_eps', type=float, default=None)


def clip_grad(single_example_batch_grad, l2_norm_clip):
    """Clip the norm of a gradient for a single-example batch."""
    grads, tree_def = jax.tree_util.tree_flatten(single_example_batch_grad)
    total_grad_norm = jnp.linalg.norm(jnp.array(
        [jnp.linalg.norm(grad.ravel()) for grad in grads]))
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_grads = [g / divisor for g in grads]
    clip_mask = jnp.asarray(total_grad_norm > l2_norm_clip, int)
    return jax.tree_util.tree_unflatten(tree_def, normalized_grads), total_grad_norm, clip_mask


@partial(jax.jit, static_argnums=(5,))
def noise_and_normalize(grad_sum, l2_norm_clip, noise_multiplier, n, prng_key, save_noise_tree):
    grad_sum_flat, grad_sum_treedef = jax.tree_util.tree_flatten(grad_sum)
    # Add DP noise; divide by batch size
    rngs = jax.random.split(prng_key, len(grad_sum_flat))

    if not save_noise_tree:
        normalized_noised_grad_sum = [
            (g + l2_norm_clip * noise_multiplier * jax.random.normal(r, g.shape)) / n
            for r, g in zip(rngs, grad_sum_flat)]
        noises_tree = None
    else:
        normalized_noised_grad_sum = []
        noises = []
        for r, g in zip(rngs, grad_sum_flat):
            noise = l2_norm_clip * noise_multiplier * jax.random.normal(r, g.shape)
            noises.append(noise)
            normalized_noised_grad_sum.append((g + noise) / n)
        noises_tree = jax.tree_util.tree_unflatten(grad_sum_treedef, noises)

    normalized_noised_grad_sum = jax.tree_util.tree_unflatten(grad_sum_treedef, normalized_noised_grad_sum)

    return normalized_noised_grad_sum, noises_tree


class DPIterativeTrainer(IterativeTrainer):
    def __init__(
            self,
            conf: configlib.Config,
            model_fn,
            train_set,
            test_set,
            seed: int,
    ):
        super().__init__(conf, model_fn, train_set, test_set, seed)
        if self.conf.priv_accountant == 'rdp':
            self.privacy_accountant = RDPAccountant()
        elif self.conf.priv_accountant == 'prv':
            self.privacy_accountant = PRVAccountant()
        else:
            raise NotImplementedError('Unknown privacy accountant.')

        if self.conf.target_eps is not None:
            self.noise_multiplier = get_noise_multiplier(
                target_epsilon=self.conf.target_eps,
                target_delta=1e-5,
                sample_rate=1 / len(self.train_loader),
                epochs=self.conf.epochs,
                accountant=self.privacy_accountant.mechanism(),
            )
        else:
            self.noise_multiplier = self.conf.dp_noise_multiplier

    def loader(self, *args, shuffle=True, drop_last=True, **kwargs):
        loader = super().loader(*args, **kwargs)
        if shuffle and self.conf.dp_sampling:
            loader = DPDataLoader.from_data_loader(loader)
            loader = DPDataLoader.from_data_loader(loader)
        return loader

    @property
    def value_and_clipped_grad(self):
        if hasattr(self, '_value_and_clipped_grad'):
            return self._value_and_clipped_grad

        @jax.jit
        def _value_and_clipped_grad(theta, X, y, l2_norm_clip):
            loss, grad = self.compute_grads(theta, X, y)
            grad, total_grad_norm, clip_mask = clip_grad(grad, l2_norm_clip)
            return loss, grad, total_grad_norm, clip_mask

        self._value_and_clipped_grad = _value_and_clipped_grad
        return self._value_and_clipped_grad

    @property
    def dp_grads(self):
        if hasattr(self, '_dp_grads'):
            return self._dp_grads

        @partial(jax.jit, static_argnums=(4,))
        def _dp_grads(theta, X, y, l2_norm_clip, save_clip_unclip):
            # Get clipped, per example gradients.
            X = jnp.expand_dims(X, axis=1)
            y = jnp.expand_dims(y, axis=1)

            per_example_loss, per_example_clipped_grads, total_grad_norm, clip_mask = \
                jax.vmap(self.value_and_clipped_grad, in_axes=(None, 0, 0, None))(
                    theta, X, y, l2_norm_clip)

            # Sum grads
            clipped_grads_flat, grads_treedef = jax.tree_util.tree_flatten(per_example_clipped_grads)
            aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
            sum_dp_grads = jax.tree_util.tree_unflatten(grads_treedef, aggregated_clipped_grads)
            # sum_dp_grads = jax.tree_map(lambda x: x.sum(axis=0), per_example_clipped_grads)

            sum_dp_grads_only_clipped, sum_dp_grads_only_unclipped = None, None
            std_angle_clipped = std_angle_unclipped = jnp.nan
            if save_clip_unclip:
                # sum of only clipped/unclipped grads
                dp_grads_only_clipped = []
                dp_grads_only_unclipped = []
                for g in clipped_grads_flat:
                    fake_ones = jnp.ones(g.shape)
                    tmp_mask_clipped = multiply_along_axis(fake_ones, clip_mask, axis=0)
                    tmp_mask_unclipped = multiply_along_axis(fake_ones, (1 - clip_mask), axis=0)
                    tmp_clipped_grad = jnp.multiply(g, tmp_mask_clipped)
                    tmp_unclipped_grad = jnp.multiply(g, tmp_mask_unclipped)
                    dp_grads_only_clipped.append(tmp_clipped_grad.sum(0))
                    dp_grads_only_unclipped.append(tmp_unclipped_grad.sum(0))
                sum_dp_grads_only_clipped = jax.tree_util.tree_unflatten(grads_treedef, dp_grads_only_clipped)
                sum_dp_grads_only_unclipped = jax.tree_util.tree_unflatten(grads_treedef, dp_grads_only_unclipped)

            return per_example_clipped_grads, per_example_loss, sum_dp_grads, \
                   sum_dp_grads_only_clipped, sum_dp_grads_only_unclipped, \
                   clip_mask, total_grad_norm

        self._dp_grads = _dp_grads
        return self._dp_grads

    def compute_update(self, theta, X, y, metadata={}, *kwargs):
        batch_size = X.shape[0]

        save_clip_unclip = save_noise_tree = False

        if self.conf.virtual_batch_size is None:
            # virtual_batch_size = self.conf.batch_size
            virtual_batch_size = batch_size
        else:
            virtual_batch_size = self.conf.virtual_batch_size

        # NOTE: for some reason this loop is much faster than jax.lax.map
        virtual_batch_num = math.ceil(batch_size / virtual_batch_size)
        grads, dp_grads_only_clipped, dp_grads_only_unclipped = None, None, None
        loss = 0.
        num_clipped = 0
        grad_norm_list = jnp.array([])
        std_angle_clipped_list = []
        std_angle_unclipped_list = []
        for i in range(virtual_batch_num):
            start = i * virtual_batch_size
            end = (i + 1) * virtual_batch_size
            per_example_clipped_grads, losses, _grads, _dp_grads_only_clipped, _dp_grads_only_unclipped, \
                clip_mask, grad_norms = self.dp_grads(
                    theta, X[start:end], y[start:end],
                    l2_norm_clip=self.conf.dp_l2_norm_clip, save_clip_unclip=save_clip_unclip
                )
            loss += float(jnp.sum(losses))
            num_clipped += jnp.sum(clip_mask)
            grad_norm_list = jnp.concatenate([grad_norm_list, grad_norms])

            if grads is None:
                grads = _grads
                dp_grads_only_clipped = _dp_grads_only_clipped
                dp_grads_only_unclipped = _dp_grads_only_unclipped
            else:
                grads = jax.tree_util.tree_map(jnp.add, grads, _grads)
                dp_grads_only_clipped = jax.tree_util.tree_map(jnp.add, dp_grads_only_clipped, _dp_grads_only_clipped)
                dp_grads_only_unclipped = jax.tree_util.tree_map(jnp.add, dp_grads_only_unclipped,
                                                                 _dp_grads_only_unclipped)

        if 'clean_grad' in self.collect_interm_out:
            clean_grad = jax.jit(jax.grad(self.forward))(self.theta, X, y)  # sgd grads
            self.metrics.interm_parts['clean_grad'] = clean_grad
        if 'clipped_grad' in self.collect_interm_out:
            self.metrics.interm_parts['clipped_grad'] = grads
        if 'dp_grads_only_clipped' in self.collect_interm_out:
            self.metrics.interm_parts['dp_grads_only_clipped'] = dp_grads_only_clipped
        if 'dp_grads_only_unclipped' in self.collect_interm_out:
            self.metrics.interm_parts['dp_grads_only_unclipped'] = dp_grads_only_unclipped
            metadata["num_clipped"] = int(num_clipped)
        if not self.conf.debug and 'grad_norm' in self.collect_interm_out:
            if (metadata['epoch_step'] == metadata['steps_per_epoch'] - 1) and (metadata['epoch'] % 10 == 0):
                pickle.dump(
                    grad_norm_list,
                    open(os.path.join(self.conf.res_path, 'grad_norm_epoch_{}.pkl'.format(metadata['epoch'])), 'wb'))

        grads, noise_tree = noise_and_normalize(
            grads, self.conf.dp_l2_norm_clip, self.noise_multiplier, batch_size, next(self.prng_seq),
            save_noise_tree=save_noise_tree
        )

        if 'noise_tree' in self.collect_interm_out:
            self.metrics.interm_parts['noise_tree'] = noise_tree

        if 'noised_grad' in self.collect_interm_out:
            noised_grad = grads
            self.metrics.interm_parts['noised_grad'] = noised_grad

        metadata["loss"] = float(loss / batch_size)

        return grads

    def step(self, metadata={}, *kwargs):
        X, y = next(self.train_loader_itr)
        y = jax.nn.one_hot(y, 10)

        # augmentation multiplicity
        if self.conf.augmult_num > 0:
            # FIXME: might be slow
            image_size = X[0].shape
            X_augmult = []
            y_augmult = []
            for i in range(X.shape[0]):
                tmp_X, tmp_y = apply_augmult(
                    X[i], y[i], image_size=image_size, augmult=self.conf.augmult_num, random_flip=True,
                    random_crop=True, crop_size=image_size, pad=4
                )
                X_augmult.append(tmp_X)
                y_augmult.append(tmp_y)
            X_augmult = np.stack(X_augmult, axis=0)
            y_augmult = np.stack(y_augmult, axis=0)

            # image with shape [NKHWC] (K is augmult), reshape to [N*K HWC]
            X = X_augmult.reshape((-1,) + X_augmult.shape[2:])
            y = y_augmult.reshape((-1,) + y_augmult.shape[2:])

        update = self.compute_update(self.theta, X, y, metadata=metadata)

        self.apply_update(update, metadata)

        metadata['learning_rate'] = self.opt_state.hyperparams['learning_rate']

        # Privacy Accountant
        self.privacy_accountant.step(noise_multiplier=self.noise_multiplier,
                                     sample_rate=1 / len(self.train_loader))
        eps = self.privacy_accountant.get_epsilon(delta=self.conf.delta)
        metadata['eps'] = eps

        return metadata
