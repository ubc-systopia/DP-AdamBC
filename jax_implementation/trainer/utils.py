import jax
import jax.numpy as jnp
import math
import haiku as hk

@jax.jit
def grad_norm(grad):
    grad, tree_def = jax.tree_util.tree_flatten(grad)
    return jnp.linalg.norm(jnp.array([jnp.linalg.norm(g.ravel()) for g in grad]))


@jax.jit
def dot_prod(tree_x, tree_y):
    dot = jax.tree_map(lambda g, z: jnp.dot(jnp.reshape(g, (1, -1)), jnp.reshape(z, (-1, 1))), tree_x, tree_y)
    return jax.tree_util.tree_reduce(lambda agg, x: agg + x, dot, 0)


def get_first(xs):
    """Gets values from the first device."""
    return jax.tree_map(lambda x: x[0], xs)


def multiply_along_axis(A, B, axis):
    return jnp.swapaxes(jnp.swapaxes(A, axis, -1) * B, -1, axis)


def _calculate_angle(vec_x, vec_y):
    # input vec_x, vec_y can be in dtype of 'pytree'
    if vec_x is None or vec_y is None:
        return jnp.nan
    else:
        if len(vec_x.keys()) != len(vec_y.keys()):
            common_keys = vec_x.keys() & vec_y.keys()
            _, vec_x = hk.data_structures.partition(lambda m, n, p: m not in list(common_keys), vec_x)
            _, vec_y = hk.data_structures.partition(lambda m, n, p: m not in list(common_keys), vec_y)
        dot_x_y = dot_prod(vec_x, vec_y)
        angle_x_y = jnp.arccos(dot_x_y / (grad_norm(vec_x) * grad_norm(vec_y)))
        return angle_x_y.flatten()  # fixme: add assert to check shape


def tree_zeros_like(tree):
    def f(x):
        return jnp.zeros_like(x)
    return jax.tree_map(f, tree)


def tree_ones_like(tree):
    def f(x):
        return jnp.ones_like(x)
    return jax.tree_map(f, tree)


def tree_flatten_1dim(tree):
    tree_flat, _ = jax.tree_util.tree_flatten(tree)
    return jnp.concatenate([i.flatten() for i in tree_flat])
