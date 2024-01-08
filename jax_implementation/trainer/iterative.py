from typing import Any, Generator, Optional, NamedTuple
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from functools import partial
from data_utils.jax_dataloader import NumpyLoader, Cycle
from tqdm import tqdm
import time
import pickle
import configlib
import os
import numpy as np
from trainer.utils import grad_norm, tree_flatten_1dim


parser = configlib.add_parser("Trainer config")
parser.add_argument("--epochs", type=int,
        help="The number of epochs to train for (if steps not set).")
parser.add_argument("--steps", type=int,
        help="The number of steps to train for (overrides epochs).")
parser.add_argument("--batch_size", "-b", default=128, type=int,
        help="The data loader batch size.")
parser.add_argument("--learning_rate", "--lr", default=.1, type=float,
        help="The leraning rate to use (or start/max on schedules).")
parser.add_argument("--cosine_lr", default=False, action='store_true',
        help="Use a one cycle cosine leraning rate schedule.")
parser.add_argument("--ema", default=False, action='store_true',
        help="Use exponential moving average parameters for evaluation.")
parser.add_argument("--optimizer", default='sgd', choices=['sgd', 'adam'])


def save_ckpt(ckpt_dir: str, params, state, idx) -> None:
    with open(os.path.join(ckpt_dir, "param_{}.pkl".format(idx)), "wb") as f:
        pickle.dump(params, f)
    with open(os.path.join(ckpt_dir, "state_{}.pkl".format(idx)), "wb") as f:
        pickle.dump(state, f)


def restore_ckpt(ckpt_dir, load_idx):
    with open(os.path.join(ckpt_dir, "param_{}.pkl".format(load_idx)), "rb") as f:
        loaded_params = pickle.load(f)
    with open(os.path.join(ckpt_dir, "state_{}.pkl".format(load_idx)), "rb") as f:
        loaded_state = pickle.load(f)
    return loaded_params, loaded_state


parser.add_argument("--n_data_workers", type=int, default=0,
        help="The number of workers to give the data loader.")


class TrainerGen(object):
    """A generator wrapper that captures the reutrn value,
    and exposes the length"""
    gen: Generator
    value: Any = None

    def __init__(self, gen: Generator, length: int):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        self.value = yield from self.gen


# An example iterative algorithm
class IterativeTrainer(object):
    def __init__(
        self,
        conf: configlib.Config,
        model_fn,
        train_set,
        test_set,
        seed: int,
    ):
        self.conf = conf
        self.model_fn = model_fn

        self.train_set = self.preprocess_dataset(train_set, train=True)
        self.train_loader = self.loader(self.train_set)
        self.train_loader_itr = iter(Cycle(self.train_loader))
        self.test_set = self.preprocess_dataset(test_set, train=False)
        self.test_loader = self.loader(self.test_set, shuffle=False, drop_last=False)

        self.prng_seq = hk.PRNGSequence(seed)

        self.net = hk.without_apply_rng(hk.transform(model_fn))

        res = next(iter(self.train_loader))
        if conf.reload_ckpt_path is None:
            self.theta = self.net.init(next(self.prng_seq), res[0])
        else:
            loaded_params, loaded_state = restore_ckpt(ckpt_dir=conf.reload_ckpt_path, load_idx=conf.reload_ckpt_idx)
            self.theta = loaded_params
            self.avg_theta = None
            self.opt_state_restored = loaded_state

        self.setup_optimizer()

        self.global_step = 0

        self.best_test_accuracy = 0.0

    def ema_update(self, params, avg_params):
        return optax.incremental_update(params, avg_params, step_size=0.0001)

    def preprocess_dataset(self, dataset, train):
        return dataset

    @property
    def lr(self):
        lr = self.conf.learning_rate
        if self.conf.cosine_lr:
            assert(self.conf.steps is not None or self.conf.epochs is not None)
            steps = self.conf.steps or self.conf.epochs * len(self.train_loader)
            lr = optax.cosine_decay_schedule(
                init_value=self.conf.learning_rate,
                decay_steps=steps,
            )
        return lr

    def setup_optimizer(self):
        if self.conf.optimizer == 'sgd':
            self.opt = optax.inject_hyperparams(optax.sgd)(learning_rate=self.lr, momentum=self.conf.beta_1)
        elif self.conf.optimizer == 'adam':
            self.opt = optax.inject_hyperparams(optax.adam)(
                learning_rate=self.lr, b1=self.conf.beta_1, b2=self.conf.beta_2,
            )
        else:
            raise NotImplementedError

        if self.conf.reload_ckpt_path is None:
            self.opt_state = self.opt.init(self.theta)
        else:
            self.opt_state = self.opt_state_restored

    def loader(self, dataset, shuffle=True, drop_last=True, batch_size=None):
        if batch_size is None:
            batch_size = self.conf.batch_size

        loader = NumpyLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            #  pin_memory=True,
            num_workers=self.conf.n_data_workers,
        )
        return loader

    def step(self, metadata={}, *kwargs):
        X, y = next(self.train_loader_itr)
        y = jax.nn.one_hot(y, 10)

        update = self.compute_update(self.theta, X, y, metadata=metadata)

        self.apply_update(update, metadata)

        metadata['learning_rate'] = self.opt_state.hyperparams['learning_rate']

        return metadata

    def compute_update(self, theta, X, y, metadata={}, *kwargs):
        # rng, self.rng = jax.random.split(self.rng)
        loss, grads = self.compute_grads(theta, X, y)
        metadata["loss"] = float(loss)

        return grads

    def apply_update(self, update, metadata={}, *kwargs):
        metadata['update_norm'] = float(grad_norm(update))
        update, self.opt_state = self.opt.update(update, self.opt_state)

        self.theta = optax.apply_updates(self.theta, update)

        if self.conf.ema:
            self.avg_theta = self.ema_update(self.theta, self.avg_theta)


    @property
    def forward(self):
        if hasattr(self, '_forward'):
            return self._forward

        @jax.jit
        def _forward(theta, X, y):
            y_hat = self.net.apply(theta, X)
            return jnp.mean(optax.softmax_cross_entropy(y_hat, y))

        self._forward = _forward
        return self._forward

    @property
    def compute_grads(self):
        if hasattr(self, '_compute_grads'):
            return self._compute_grads

        self._compute_grads = jax.jit(jax.value_and_grad(self.forward, argnums=0))

        return self._compute_grads

    @property
    def correct_preds(self):
        if hasattr(self, '_correct_preds'):
            return self._correct_preds

        @jax.jit
        def _correct_preds(theta, X, y):
            batch_size = X.shape[0]
            y_onehot = jax.nn.one_hot(y, 10)
            y = jnp.expand_dims(y, axis=-1)

            y_hat = self.net.apply(theta, X)
            test_loss = jnp.sum(optax.softmax_cross_entropy(y_hat, y_onehot))
            y_hat = jnp.expand_dims(jnp.argmax(y_hat, axis=1), axis=-1)
            correct = jnp.sum(y_hat == y)
            return correct, batch_size, test_loss

        self._correct_preds = _correct_preds
        return self._correct_preds

    def _do_step(self):
        """Updates relevant metadata, and calls step().
        """
        self.steps_per_epoch = len(self.train_loader)
        self.epoch = self.global_step // self.steps_per_epoch
        self.epoch_step = self.global_step % self.steps_per_epoch

        res = self.step(metadata={
            'epoch': self.epoch,
            'steps_per_epoch': self.steps_per_epoch,
            'step': self.global_step,
            'epoch_step': self.epoch_step,
        })

        self.global_step += 1

        return res

    def _train_iter(self, steps):
        assert(steps>=0)
        for step in range(steps):
            yield self._do_step()

    def train_iter(self, epochs=None, steps=None):
        if steps is None and epochs is not None:
            steps = epochs * len(self.train_loader)

        if steps is None:
            assert(self.conf.steps is not None or self.conf.epochs is not None)
            steps = self.conf.steps or self.conf.epochs * len(self.train_loader)

        return TrainerGen(self._train_iter(steps), length=steps)

    def train(self, *args, **kwargs):
        gen = self.train_iter(*args, **kwargs)
        _ = tuple(gen)

    def _eval_iter(self):
        correct = tot = 0
        test_loss = 0
        for data in self.test_loader:
            if self.conf.ema:
                _correct, _tot, _test_loss = self.correct_preds(self.avg_theta, data[0], data[1])
            else:
                _correct, _tot, _test_loss = self.correct_preds(self.theta, data[0], data[1])

            correct += int(_correct)
            tot += int(_tot)
            test_loss += float(_test_loss)

            yield correct / tot, test_loss / tot

        cur_test_accuracy = correct / tot
        if self.conf.save_mode:
            save_ckpt(ckpt_dir=self.conf.res_path, params=self.theta, state=self.opt_state, idx='last')

        if cur_test_accuracy > self.best_test_accuracy:
            self.best_test_accuracy = cur_test_accuracy
            if self.conf.save_mode:
                save_ckpt(ckpt_dir=self.conf.res_path, params=self.theta, state=self.opt_state, idx='best')

        return cur_test_accuracy, test_loss / tot, self.best_test_accuracy

    def eval_iter(self):
        return TrainerGen(self._eval_iter(), length=len(self.test_loader))

    def eval(self):
        gen = self.eval_iter()
        _ = tuple(gen)
        return gen.value

