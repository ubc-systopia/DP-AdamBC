import functools

from trainer.dp_iterative import DPIterativeTrainer, noise_and_normalize, clip_grad
from trainer.utils import tree_zeros_like, tree_flatten_1dim, grad_norm, tree_ones_like
from data_utils.jax_dataloader import NumpyLoader, Cycle

import configlib
import math
import jax
import jax.numpy as jnp
from functools import partial
from itertools import chain
from opacus.accountants.rdp import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
import pickle
import os
import optax
from optax._src import base
from optax._src import utils
from optax._src import numerics
from optax._src import combine
from optax._src.transform import ScaleByAdamState, update_moment, update_moment_per_elem_norm, bias_correction, TraceState
# from optax._src.transform import ScaleByAdamState, _update_moment, _update_moment_per_elem_norm, _bias_correction, TraceState
from optax._src.alias import _scale_by_learning_rate
from typing import Any, Callable, NamedTuple, Optional, Union
import chex

parser = configlib.add_parser("DP-Adam Trainer config")
parser.add_argument("--beta_1", default=0.9, type=float)
parser.add_argument("--beta_2", default=0.999, type=float)
parser.add_argument("--adam_corr", default=False, action='store_true')
parser.add_argument("--adam_corr_after_epoch", default=-1, type=int)
parser.add_argument("--sgd_momentum", default=False, action='store_true')
parser.add_argument("--imp_min", type=float, default=0)
parser.add_argument("--imp_max", type=float, default=1)
parser.add_argument("--tmp_bias", type=float, default=0)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--eps_root", type=float, default=1e-8)
parser.add_argument("--dict_path", type=str, default="/tmp")
parser.add_argument("--clipping_only", default=False, action='store_true')
parser.add_argument("--gamma_decay", type=float, default=1.0)
parser.add_argument("--lr_decay", type=float, default=1.0)


def _gamma_exponential_decay_scheduler(step_number, init_value, decay_rate):
    decayed_value = init_value * (decay_rate ** (step_number / 1))
    return decayed_value


class ScaleByAdamStateCorr(NamedTuple):
    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: base.Updates
    nu: base.Updates
    nu_corr: base.Updates
    count_tree: None  # individual param record of update count


class ScaleByAdamStateCorrLong(NamedTuple):
    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: base.Updates
    nu: base.Updates
    nu_corr: base.Updates
    mu_clean: base.Updates
    nu_clean: base.Updates
    perc_corr: None


def scale_by_adam_corr(
        batch_size: int,
        dp_noise_multiplier: float,
        dp_l2_norm_clip: float,
        b1: float,
        b2: float,
        eps: float,
        eps_root: float,
        eps_root_decay: float,
        mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        nu_corr = jax.tree_util.tree_map(jnp.zeros_like, params)  # corrected second moment
        mu_clean = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        nu_clean = jax.tree_util.tree_map(jnp.zeros_like, params)
        return ScaleByAdamStateCorrLong(
            count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, nu_corr=nu_corr,
            mu_clean=mu_clean, nu_clean=nu_clean, perc_corr=None,
        )

    def update_fn(updates, state, params=None):
        del params
        noised_updates, clean_updates = updates
        mu = update_moment(noised_updates, state.mu, b1, 1)
        nu = update_moment_per_elem_norm(noised_updates, state.nu, b2, 2)
        count_inc = numerics.safe_int32_increment(state.count)
        mu_hat = bias_correction(mu, b1, count_inc)
        # nu_hat_uncorr = bias_correction(nu, b2, count_inc)
        # corr for noise variance
        nu_hat = bias_correction(nu, b2, count_inc)
        noise_err = (1 / batch_size ** 2) * dp_noise_multiplier ** 2 * dp_l2_norm_clip ** 2
        # without gamma scheduler
        nu_hat_corr = jax.tree_map(lambda x: jnp.maximum(x - noise_err, eps_root), nu_hat)
        # # # 1- replace small values with eps_root
        # nu_corr = jax.tree_map(lambda x: jnp.maximum(x - noise_err, eps_root), nu)
        # nu_hat = bias_correction(nu_corr, b2, count_inc)
        # nu_corr_orig = jax.tree_map(lambda x: jnp.maximum(x - noise_err, 1e-30), nu)
        # nu_hat_corr_orig = bias_correction(nu_corr_orig, b2, count_inc)
        # updates = jax.tree_util.tree_map(
        #     lambda m, v: m / (jnp.sqrt(v) + eps), mu_hat, nu_hat)
        # updates = jax.tree_util.tree_map(
        #     lambda m, v: m / (jnp.sqrt(v)), mu_hat, nu_hat)
        updates = jax.tree_util.tree_map(lambda m, v: m / (jnp.sqrt(v)), mu_hat, nu_hat_corr)
        num_corr1 = jnp.sum(tree_flatten_1dim(jax.tree_map(lambda x: jnp.sum((x - noise_err) > eps_root), nu_hat)))
        num_corr2 = 0
        # # 3- max(v_corr, m_t^2) + gamma
        # nu_corr = jax.tree_map(lambda x: x - noise_err, nu)
        # nu_hat = bias_correction(nu_corr, b2, count_inc)
        # tmp_square_mu = jax.tree_map(jnp.square, mu_hat)
        # nu_corr2 = jax.tree_map(lambda x, y: jnp.add(jnp.maximum(x, y), eps_root), nu_hat, tmp_square_mu)
        # nu_corr2 = jax.tree_map(lambda x, y: jnp.maximum(x, y), nu_hat, tmp_square_mu)
        # updates = jax.tree_util.tree_map(lambda m, v: m / (jnp.sqrt(v)), mu_hat, nu_corr2)
        # num_corr1 = jnp.sum(tree_flatten_1dim(jax.tree_map(lambda x, y: jnp.sum(x > y), nu_hat, tmp_square_mu)))
        # num_corr2 = jnp.sum(tree_flatten_1dim(jax.tree_map(lambda x: x == eps_root, nu_corr2)))
        mu = utils.cast_tree(mu, mu_dtype)

        # clean states updated using clipped grads
        mu_clean = update_moment(clean_updates, state.mu_clean, b1, 1)
        nu_clean = update_moment_per_elem_norm(clean_updates, state.nu_clean, b2, 2)
        mu_hat_clean = bias_correction(mu_clean, b1, count_inc)
        nu_hat_clean = bias_correction(nu_clean, b2, count_inc)

        # logging
        dummy_count = jnp.sum(tree_flatten_1dim(jax.tree_map(lambda x: jnp.sum(~jnp.isnan(x)), nu_hat)))
        perc_corr1 = num_corr1 / dummy_count
        perc_corr2 = 1 - (num_corr2 / dummy_count)

        return updates, ScaleByAdamStateCorrLong(
            count=count_inc, mu=mu, nu=nu,
            nu_corr=nu_hat_corr,
            mu_clean=mu_clean, nu_clean=nu_clean, perc_corr=(perc_corr1, perc_corr2),
        )

    return base.GradientTransformation(init_fn, update_fn)


def scale_by_adam_orig(
        b1: float,
        b2: float,
        eps: float,
        tmp_bias: float,
        mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        nu_corr = jax.tree_util.tree_map(jnp.zeros_like, params)  # corrected second moment
        mu_clean = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        nu_clean = jax.tree_util.tree_map(jnp.zeros_like, params)
        return ScaleByAdamStateCorrLong(
            count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, nu_corr=nu_corr,
            mu_clean=mu_clean, nu_clean=nu_clean, perc_corr=None,
        )

    def update_fn(updates, state, params=None):
        del params
        noised_updates, clean_updates = updates
        mu = update_moment(noised_updates, state.mu, b1, 1)
        nu = update_moment_per_elem_norm(noised_updates, state.nu, b2, 2)
        count_inc = numerics.safe_int32_increment(state.count)
        mu_hat = bias_correction(mu, b1, count_inc)
        nu_hat = bias_correction(nu, b2, count_inc)
        updates = jax.tree_util.tree_map(
            lambda m, v: m / (jnp.sqrt(v) + eps), mu_hat, nu_hat)
        # # tmp: ablation study purposes
        # updates = mu_hat
        mu = utils.cast_tree(mu, mu_dtype)

        # clean states updated using clipped grads
        mu_clean = update_moment(clean_updates, state.mu_clean, b1, 1)
        nu_clean = update_moment_per_elem_norm(clean_updates, state.nu_clean, b2, 2)
        mu_hat_clean = bias_correction(mu_clean, b1, count_inc)
        nu_hat_clean = bias_correction(nu_clean, b2, count_inc)

        # tmp_bias: ablation study purposes
        # updates = jax.tree_util.tree_map(
        #     lambda m, v: m / (jnp.sqrt(v + tmp_bias) + eps), mu_hat, nu_hat_clean)
        # updates = jax.tree_util.tree_map(
        #     lambda m, v: m / (jnp.sqrt(v + tmp_bias) + eps), mu_hat, nu_hat)

        return updates, ScaleByAdamStateCorrLong(
            count=count_inc, mu=mu, nu=nu, nu_corr=None,
            mu_clean=mu_clean, nu_clean=nu_clean, perc_corr=(-1, -1),
        )

    return base.GradientTransformation(init_fn, update_fn)


def adam(
    batch_size: int,
    dp_noise_multiplier: float,
    dp_l2_norm_clip: float,
    learning_rate: float,
    b1: float,
    b2: float,
    eps: float,
    eps_root: float,
    eps_root_decay: float,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    return combine.chain(
        scale_by_adam_corr(
            batch_size=batch_size, dp_noise_multiplier=dp_noise_multiplier, dp_l2_norm_clip=dp_l2_norm_clip,
            b1=b1, b2=b2, eps=eps, eps_root=eps_root, eps_root_decay=eps_root_decay, mu_dtype=mu_dtype),
        _scale_by_learning_rate(learning_rate),
    )


def adam_orig(
    learning_rate: float,
    b1: float,
    b2: float,
    eps: float,
    tmp_bias: float,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    return combine.chain(
        scale_by_adam_orig(b1=b1, b2=b2, eps=eps, mu_dtype=mu_dtype, tmp_bias=tmp_bias),
        _scale_by_learning_rate(learning_rate),
    )


class MyTraceState(NamedTuple):
    """Holds an aggregation of past updates."""
    trace: base.Params
    count: chex.Array


def my_trace(
        decay: float,
        nesterov: bool = False,
        accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)

    def init_fn(params):
        return MyTraceState(
            trace=jax.tree_util.tree_map(
                lambda t: jnp.zeros_like(t, dtype=accumulator_dtype), params),
            count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None):
        del params
        f = lambda g, t: g + decay * t
        new_trace = jax.tree_util.tree_map(f, updates, state.trace)
        updates = (
            jax.tree_util.tree_map(f, updates, new_trace) if nesterov
            else new_trace)
        new_trace = utils.cast_tree(new_trace, accumulator_dtype)
        count_inc = numerics.safe_int32_increment(state.count)
        updates = bias_correction(updates, decay, count_inc)
        return updates, MyTraceState(trace=new_trace, count=count_inc)

    return base.GradientTransformation(init_fn, update_fn)


def my_sgd(
        learning_rate,
        momentum: Optional[float] = None,
        nesterov: bool = False,
        accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    return combine.chain(
        (my_trace(decay=momentum, nesterov=nesterov, accumulator_dtype=accumulator_dtype)
         if momentum is not None else base.identity()),
        _scale_by_learning_rate(learning_rate)
    )


class DPAdamTrainer(DPIterativeTrainer):
    def __init__(
            self,
            conf: configlib.Config,
            model_fn,
            train_set,
            test_set,
            seed: int,
    ):
        super().__init__(conf, model_fn, train_set, test_set, seed)
        # helper loader for measuring sampling noise
        self.helper_loader = self.loader(self.train_set)
        self.helper_loader_itr = iter(Cycle(self.helper_loader))
        self.setup_optimizer_adam()  # overwrite default optimizer

    def setup_optimizer_adam(self):
        beta_1 = self.conf.beta_1
        # beta_2 = 1 - ((1-self.conf.beta_1) ** 2)  # (1-b1) = sqrt(1-b2)
        beta_2 = self.conf.beta_2

        if self.conf.adam_corr:
            self.opt = optax.inject_hyperparams(adam)(
                batch_size=self.conf.batch_size, dp_noise_multiplier=self.noise_multiplier,
                dp_l2_norm_clip=self.conf.dp_l2_norm_clip,
                learning_rate=self.lr, b1=beta_1, b2=beta_2,
                eps=self.conf.eps, eps_root=self.conf.eps_root, eps_root_decay=-1,
            )
        elif self.conf.sgd_momentum:
            # self.opt = optax.inject_hyperparams(optax.sgd)(learning_rate=self.lr, momentum=self.conf.beta_1)
            self.opt = optax.inject_hyperparams(my_sgd)(learning_rate=self.lr, momentum=self.conf.beta_1)
        else:
            self.opt = optax.inject_hyperparams(optax.adam)(
                learning_rate=self.lr, b1=self.conf.beta_1, b2=self.conf.beta_2, eps_root=self.conf.tmp_bias,
            )
            # self.opt = optax.inject_hyperparams(adam_orig)(
            #     learning_rate=self.lr, b1=beta_1, b2=beta_2, eps=self.conf.eps, tmp_bias=self.conf.tmp_bias,
            # )

        if self.conf.reload_ckpt_path is None:
            self.opt_state = self.opt.init(self.theta)
        else:
            self.opt_state = self.opt_state_restored

    def compute_update(self, theta, X, y, metadata={}, *kwargs):
        batch_size = X.shape[0]

        if self.conf.virtual_batch_size is None:
            virtual_batch_size = batch_size
        else:
            virtual_batch_size = self.conf.virtual_batch_size

        virtual_batch_num = math.ceil(batch_size / virtual_batch_size)
        grads = None
        loss = 0.
        for i in range(virtual_batch_num):
            start = i * virtual_batch_size
            end = (i + 1) * virtual_batch_size
            per_example_clipped_grads, losses, _grads, _dp_grads_only_clipped, _dp_grads_only_unclipped, \
                clip_mask, grad_norms = self.dp_grads(
                    theta, X[start:end], y[start:end],
                    l2_norm_clip=self.conf.dp_l2_norm_clip, save_clip_unclip=False
                )
            loss += float(jnp.sum(losses))

            if grads is None:
                grads = _grads
            else:
                grads = jax.tree_util.tree_map(jnp.add, grads, _grads)

        mean_clipped_grads = jax.tree_map(lambda x: x / batch_size, grads)

        if not self.conf.clipping_only:
            grads, noise_tree = noise_and_normalize(
                grads, self.conf.dp_l2_norm_clip, self.noise_multiplier, batch_size, next(self.prng_seq),
                save_noise_tree=False
            )
        else:
            grads = mean_clipped_grads

        metadata["loss"] = float(loss / batch_size)

        return {'grads': grads, 'clipped_grads': mean_clipped_grads}

    def apply_update(self, update, metadata={}, *kwargs):
        clipped_grads = update['clipped_grads']
        update = update['grads']
        metadata['update_norm'] = float(grad_norm(update))

        if self.conf.adam_corr:
            update, self.opt_state = self.opt.update((update, clipped_grads), self.opt_state)
            _, _, _, _, _, _, perc_corr = self.opt_state[2][0]
            metadata['perc_corr1'] = float(perc_corr[0])
            metadata['perc_corr2'] = float(perc_corr[1])
        else:
            update, self.opt_state = self.opt.update(update, self.opt_state)

        self.theta = optax.apply_updates(self.theta, update)

    def step(self, metadata={}, *kwargs):
        X, y = next(self.train_loader_itr)
        y = jax.nn.one_hot(y, 10)
        update = self.compute_update(self.theta, X, y, metadata=metadata)

        self.apply_update(update, metadata)

        metadata['learning_rate'] = self.opt_state.hyperparams['learning_rate']

        # Privacy Accountant
        self.privacy_accountant.step(noise_multiplier=self.noise_multiplier,
                                     sample_rate=self.conf.batch_size / len(self.train_loader.dataset))
        eps = self.privacy_accountant.get_epsilon(delta=self.conf.delta)
        metadata['eps'] = eps

        return metadata