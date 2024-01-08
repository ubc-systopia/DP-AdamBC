import jax
import jax.numpy as jnp
import os
import optax
from optax._src import base
from optax._src import utils
from optax._src import numerics
from optax._src import combine
from optax._src.transform import ScaleByAdamState, update_moment, update_moment_per_elem_norm, bias_correction
from optax._src.alias import _scale_by_learning_rate
from typing import Any, Callable, NamedTuple, Optional, Union
import chex

LOGGING = True
# LOGGING = False

def get_summary_stats(a, prefix):
    a_flattened = tree_flatten_1dim(a)
    a_min = jnp.min(a_flattened)
    a_max = jnp.max(a_flattened)
    a_mean = jnp.mean(a_flattened)
    a_median = jnp.median(a_flattened)
    a_q25 = jnp.quantile(a_flattened, q=0.25)
    a_q75 = jnp.quantile(a_flattened, q=0.75)
    stats = {'min': a_min, 'max': a_max, 'mean': a_mean, 'median': a_median, 'q25': a_q25, 'q75': a_q75}
    return add_prefix(stats, prefix)

def add_prefix(d, prefix):
    new_dict = {}
    for key, value in d.items():
        new_dict[prefix+'_'+key] = value
    return new_dict

def tree_flatten_1dim(tree):
    tree_flat, _ = jax.tree_util.tree_flatten(tree)
    return jnp.concatenate([i.flatten() for i in tree_flat])

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
    count_tree: None  # individual param record of update count
    mu_clean: base.Updates
    nu_clean: base.Updates
    summary_stats: dict


def scale_by_adam_corr(
        sigma: float,
        b1: float,
        b2: float,
        eps: float,
        eps_root: float,
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
        summary_stats={}
        return ScaleByAdamStateCorrLong(
            count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, nu_corr=nu_corr, count_tree=None,
            mu_clean=mu_clean, nu_clean=nu_clean, summary_stats=summary_stats,
        )

    def update_fn(updates, state, params=None):
        del params
        clean_updates, noised_updates = updates
        # update moments
        mu = update_moment(noised_updates, state.mu, b1, 1)
        nu = update_moment_per_elem_norm(noised_updates, state.nu, b2, 2)
        count_inc = numerics.safe_int32_increment(state.count)
        # do bias correction
        mu_hat = bias_correction(mu, b1, count_inc)
        nu_hat_uncorr = bias_correction(nu, b2, count_inc)
        # corr for noise variance
        # sum_i=1^t x^(t-i) = (x^t-1)/(x-1), multiply by (1-x) = 1-x^t

        # Mathias suggestion
        nu_hat = bias_correction(nu, b2, count_inc)
        noise_err = sigma ** 2
        nu_corr = jax.tree_map(lambda x: jnp.maximum(x - noise_err, eps_root), nu_hat)
        # compute updates
        updates = jax.tree_util.tree_map(
            lambda m, v: m / (jnp.sqrt(v)), mu_hat, nu_corr)

        # noise_errs = jax.tree_map(
        #     lambda sigma: sigma ** 2 * (1 - b2 ** count_inc), sigmas)
        # # # 1- replace small values with eps_root
        # nu_corr = jax.tree_map(lambda x, noise_err: jnp.maximum(x - noise_err, eps_root_multiplier * noise_err), nu, noise_errs)
        # nu_hat = bias_correction(nu_corr, b2, count_inc)
        # # compute updates
        # updates = jax.tree_util.tree_map(
        #     lambda m, v: m / (jnp.sqrt(v)), mu_hat, nu_hat)


        mu = utils.cast_tree(mu, mu_dtype)
        # clean states updated using clipped grads
        mu_clean = update_moment(clean_updates, state.mu_clean, b1, 1)
        nu_clean = update_moment_per_elem_norm(clean_updates, state.nu_clean, b2, 2)
        mu_hat_clean = bias_correction(mu_clean, b1, count_inc)
        nu_hat_clean = bias_correction(nu_clean, b2, count_inc)

        # Mathias suggestion
        num_corr = jnp.sum(tree_flatten_1dim(jax.tree_map(lambda x: jnp.sum((x - noise_err) > (eps_root)), nu_hat)))


        # num_corr = jnp.sum(tree_flatten_1dim(jax.tree_map(lambda x, noise_err: jnp.sum((x - noise_err) > (eps_root_multiplier * noise_err)), nu, noise_errs)))
        dummy_count = jnp.sum(tree_flatten_1dim(jax.tree_map(lambda x: jnp.sum(~jnp.isnan(x)), nu_hat)))
        perc_corr = num_corr / dummy_count

        # perform logging
        if LOGGING:
            summary_stats =  {'perc_corr': perc_corr,
                              **get_summary_stats(mu_hat_clean, 'mt_clean'), 
                              **get_summary_stats(mu_hat, 'mt_noised'),
                              **get_summary_stats(nu_hat_clean, 'vt_clean'),
                              **get_summary_stats(nu_hat_uncorr, 'vt_noised'),
                              **get_summary_stats(nu_corr, 'vt_corr')}
        else:
            summary_stats = {}
        # return updates and state
        return updates, ScaleByAdamStateCorrLong(
            count=count_inc, mu=mu, nu=nu, nu_corr=nu_corr, count_tree=None,
            mu_clean=mu_clean, nu_clean=nu_clean, summary_stats=summary_stats
        )

    return base.GradientTransformation(init_fn, update_fn)

def adamcorr(
    sigma: float,
    learning_rate: float,
    b1: float,
    b2: float,
    eps: float,
    eps_root: float,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    return combine.chain(
        scale_by_adam_corr(
            sigma=sigma, b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
        _scale_by_learning_rate(learning_rate),
    )

def scale_by_adam(
        b1: float,
        b2: float,
        eps: float,
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
        summary_stats={}
        return ScaleByAdamStateCorrLong(
            count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, nu_corr=nu_corr, count_tree=None,
            mu_clean=mu_clean, nu_clean=nu_clean, summary_stats=summary_stats,
        )

    def update_fn(updates, state, params=None):
        del params
        clean_updates, noised_updates = updates
        # update moments
        mu = update_moment(noised_updates, state.mu, b1, 1)
        nu = update_moment_per_elem_norm(noised_updates, state.nu, b2, 2)
        count_inc = numerics.safe_int32_increment(state.count)
        # do bias correction
        mu_hat = bias_correction(mu, b1, count_inc)
        nu_hat = bias_correction(nu, b2, count_inc)
        # compute updates
        updates = jax.tree_util.tree_map(
            lambda m, v: m / (jnp.sqrt(v) + eps), mu_hat, nu_hat)
        # TODO: not sure what this line is for
        mu = utils.cast_tree(mu, mu_dtype)
        # clean states updated using clipped grads
        mu_clean = update_moment(clean_updates, state.mu_clean, b1, 1)
        nu_clean = update_moment_per_elem_norm(clean_updates, state.nu_clean, b2, 2)
        mu_hat_clean = bias_correction(mu_clean, b1, count_inc)
        nu_hat_clean = bias_correction(nu_clean, b2, count_inc)
        # perform logging
        if LOGGING:
            summary_stats =  {**get_summary_stats(mu_hat_clean, 'mt_clean'), 
                              **get_summary_stats(mu_hat, 'mt_noised'),
                              **get_summary_stats(nu_hat_clean, 'vt_clean'),
                              **get_summary_stats(nu_hat, 'vt_noised')}
        else:
            summary_stats = {}
        # return updates and state
        return updates, ScaleByAdamStateCorrLong(
            count=count_inc, mu=mu, nu=nu, nu_corr=None, count_tree=None,
            mu_clean=mu_clean, nu_clean=nu_clean, summary_stats=summary_stats
        )

    return base.GradientTransformation(init_fn, update_fn)

def adam(
    learning_rate: float,
    b1: float,
    b2: float,
    eps: float,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    return combine.chain(
        scale_by_adam(b1=b1, b2=b2, eps=eps, mu_dtype=mu_dtype),
        _scale_by_learning_rate(learning_rate),
    )