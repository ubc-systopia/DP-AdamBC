import optuna
import argparse
import json
import sys
import random

from typing import Sequence
from absl import app
from absl import flags
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf

# from sqlalchemy import create_engine

import train

_WORKDIR = flags.DEFINE_string(
    'workdir',
    None,
    'Directory to store model data.')
_CONFIG = config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=False)
_WANDB_PROJECT = None

def objective(trial, json_obj, group_name):
    from absl import logging

    workdir= "./tmp/" + str(group_name) + "/autotune_" + str(trial.number)

    for arg_obj in json_obj:
        # the arg_obj is either const or tuned
        if arg_obj["key_type"] == "const":
            if not (isinstance(arg_obj["value"], str) and arg_obj["value"] == ""):
                _CONFIG.value[arg_obj["key"]] = arg_obj["value"]
        else:
            # get trial value
            if arg_obj["value_type"] == "categorical":
                _CONFIG.value[arg_obj["key"]] = trial.suggest_categorical(arg_obj["key"], arg_obj["value"])
            elif arg_obj["value_type"] == "float":
                _CONFIG.value[arg_obj["key"]] =  trial.suggest_float(arg_obj["key"], arg_obj["value"][0], arg_obj["value"][1])
            elif arg_obj["value_type"] == "int":
                _CONFIG.value[arg_obj["key"]] = trial.suggest_int(arg_obj["key"], arg_obj["value"][0], arg_obj["value"][1])
            elif arg_obj["value_type"] == "log":
                _CONFIG.value[arg_obj["key"]] = trial.suggest_float(arg_obj["key"], arg_obj["value"][0], arg_obj["value"][1], log=True)
            else:
                raise RuntimeError("No value_type for argument")

    _CONFIG.value["wandb_project"] = None

    # do "main" setup
    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                        f'process_count: {jax.process_count()}')
    platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY, workdir, 'workdir')

    return train.train_and_evaluate(_CONFIG.value, workdir)


def run_best(config, json_obj, group_name, iteration):
    from absl import logging

    workdir= "./tmp/" + str(group_name) + "/run_" + str(iteration)

    # Set consts
    for arg_obj in json_obj:
        if arg_obj["key_type"] == "const":
            if not (isinstance(arg_obj["value"], str) and arg_obj["value"] == ""):
                _CONFIG.value[arg_obj["key"]] = arg_obj["value"]
    # Set tuned
    for key, value in config.items():
        _CONFIG.value[key] = value
    _CONFIG.value["experiment_name"] = "run_" + str(iteration)
    _CONFIG.value["rng_seed"] = random.randint(0, 1000000)
    _CONFIG.value["wandb_project"] = _WANDB_PROJECT

    # do "main" setup
    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                        f'process_count: {jax.process_count()}')
    platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY, workdir, 'workdir')

    train.train_and_evaluate(_CONFIG.value, workdir)


def save_best(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
    with open(f"./autotune/best_params/best_params_{study.study_name}.json", "w") as json_file:
        best_params = study.best_params
        best_params['trial_number'] = study.best_trial.number
        print(f"Saving best trial to: best_params_{study.study_name}.json")
        json.dump(best_params, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", help="input JSON for tuning", type=str)
    parser.add_argument("--n", default=10, help="number of trials", type=int)
    args, unknown_flags = parser.parse_known_args()

    flags.FLAGS(sys.argv[:1] + unknown_flags)  # Let absl.flags parse the rest.
    flags.mark_flags_as_required(['config', 'workdir'])

    _WANDB_PROJECT = _CONFIG.value["wandb_project"]
    print("wandb project name:", _WANDB_PROJECT)

    # load json
    with open(f"./autotune/{args.in_path}") as json_file:
        json_obj = json.load(json_file)

    study_name = f"{args.in_path.replace('.json', '')}_study"  # Unique identifier of the study.
    storage_name = "sqlite:///./autotune/sql/{}.db".format(study_name)
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_name, load_if_exists=True)
    print(f"Completed {len(study.trials)} / {len(study.trials)+args.n} trials")
    study.optimize(lambda trial : objective(trial, json_obj, study_name), n_trials=args.n, callbacks=[save_best])

    for i in range(5):
        run_best({}, json_obj, study_name, i)
    # for i in range(5):
    #     run_best(study.best_params, json_obj, study_name, i)