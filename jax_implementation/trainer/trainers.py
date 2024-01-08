"add trainer config params for the various trainer options available"
import configlib
from trainer.iterative import IterativeTrainer
from trainer.dp_iterative import DPIterativeTrainer
from trainer.dp_adambc import DPAdamTrainer

parser = configlib.add_parser("Trainer config")
parser.add_argument("--trainer", default="DPIterative", type=str,
    help="The trainer to use, class name (e.g. Iterative, DPIterative, ...). Trainer is appended",)


def get_trainer(conf: configlib.Config, model_fn, train_set, test_set, seed):
    trainer_class_name = f"{conf.trainer}Trainer"
    if trainer_class_name not in globals():
        raise NotImplementedError

    Trainer = globals()[trainer_class_name]
    trainer = Trainer(
        conf=conf, model_fn=model_fn, train_set=train_set, test_set=test_set, seed=seed)
    return trainer

