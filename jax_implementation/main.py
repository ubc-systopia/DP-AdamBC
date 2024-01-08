import os
import pprint
import configlib

from data_utils.datasets import get_dataset
from models.classifiers import get_classifier
from trainer.trainers import get_trainer

from trainer.iterative import IterativeTrainer
from trainer.dp_iterative import DPIterativeTrainer

import random
from tqdm import tqdm
import time
import json
import pickle
import torch
import numpy as np
from jax.config import config
import wandb

# Configuration arguments
parser = configlib.add_parser("Run config")
parser.add_argument("--gpu_id", type=int,
        help="Force the use of a specfici GPU id.")
parser.add_argument("--progress_bar", default=False, action='store_true',
        help="Enable progress bars (e.g. when training locally).")
parser.add_argument("--seed", type=int,
        help="the Trainer's jax/haiku seed. Random int in [0, 1000[ if unspecified.")
parser.add_argument("--debug", default=False, action='store_true',
                    help="save no outputs in debug mode")
parser.add_argument("--out_path", type=str, default="$HOME/tmp",
                    help="the output directory")
parser.add_argument("--exp_name", type=str, default="exp",
                    help="experiment name to save")
parser.add_argument("--reload_ckpt_path", type=str, default=None,
                    help="checkpoint path to reload")
parser.add_argument("--reload_ckpt_idx", type=str, default=None,
                    help="either best or last")
parser.add_argument("--exp_group", type=str, default=None)
parser.add_argument("--exp_proj", type=str, default=None)
parser.add_argument("--disable_jit", default=False, action='store_true')
parser.add_argument("--eval_every", type=int, default=1000)


def main(input_args=None):
    conf = configlib.parse(input_args=input_args)
    pprint.pprint(conf)

    # In debug mode turn off JIT
    if conf.disable_jit:
        config.update('jax_disable_jit', True)

    if conf.gpu_id is not None: os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_id
    if conf.seed is None:
        seed = random.randint(0, 1000)
        print('****** No seed given, generating random seed = {} ******'.format(seed))
        conf.update({'seed': seed})
    else:
        seed = conf.seed

    if not conf.debug:
        wandb.init(project=conf.exp_proj, config=conf, group=conf.exp_group, name=conf.exp_name)

    train_set, test_set = get_dataset(conf)
    model_fn = get_classifier(conf)
    trainer = get_trainer(
        conf=conf,
        model_fn=model_fn,
        train_set=train_set,
        test_set=test_set,
        seed=seed,
    )

    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    neg_loss_improvement = 0
    acc = 0
    pbar_it = tqdm(trainer.train_iter(), disable=not conf.progress_bar)
    for metadata in pbar_it:
        if metadata['step'] % conf.eval_every == 0:
            neg_loss_improvement = 0
            acc, test_loss, best_acc = trainer.eval()
            # Go to new line for progress bar
            if conf.progress_bar: print("")
            if not conf.debug:
                test_log_dict = {'test_accuracy': acc, 'best_test_accuracy': best_acc, 'test_loss': test_loss}
                test_log_dict.update({'test_step': metadata.get('step')})
                test_log_dict.update({'dp_epsilon': metadata.get('eps')})
                wandb.log(test_log_dict)

        desc = f"E={metadata['epoch']} A={100*acc:.2f} S={metadata['epoch_step']} L={metadata['loss']:.2f}"
        if 'learning_rate' in metadata:
            desc += f" LR={metadata['learning_rate']:.4f}"
        if 'loss_improvement' in metadata:
            desc += f" LI={metadata['loss_improvement']:.4f}"
            if metadata['loss_improvement'] < 0:
                neg_loss_improvement += 1
            desc += f" Ø={neg_loss_improvement}"
        if 'update_norm' in metadata:
            desc += f" ‖Δ‖={metadata['update_norm']:.2f}"
        if 'mu_norm' in metadata:
            desc += f" ‖µ‖={metadata['mu_norm']:.2f}"
        if 'diff_norm' in metadata:
            desc += f" ‖≠‖={metadata['diff_norm']:.2f}"
        if 'eps' in metadata:
            desc += f" ||≠||={metadata['eps']:.2f}"
            if conf.target_eps is not None and metadata['eps'] > conf.target_eps:
                print('Reached target epsilon..')
                break
        if 'memory_norms' in metadata:
            memory_norms = metadata["memory_norms"]
            min_norm = min(memory_norms)
            max_norm = max(memory_norms)
            avg_norm = sum(memory_norms)/len(memory_norms)
            metadata["min_norm"] = min_norm
            metadata["max_norm"] = max_norm
            metadata["avg_norm"] = avg_norm
            del metadata['memory_norms']
            desc += f" ‖mem‖={min_norm:.1f}-[{avg_norm:.1f}]-{max_norm:.1f}"
        pbar_it.set_description(desc)

        if not conf.debug:
            del metadata['metrics']  # avoid saving a python object
            wandb.log(metadata)

    if not conf.debug:
        wandb.finish()

    return acc


if __name__ == "__main__":
    main()
