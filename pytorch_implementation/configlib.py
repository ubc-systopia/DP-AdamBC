"""Lightweight configuration library supporting commmand line and json arguments."""
from typing import Dict, Any
import logging
import sys
import json
import argparse

# Logging for config library
logger = logging.getLogger(__name__)

# Our global parser that we will collect arguments into
parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS, fromfile_prefix_chars="@")

def add_parser(title: str, description: str = ""):
    """Create a new shared/global context for arguments and return a handle."""
    return parser.add_argument_group(title, description)

p = add_parser("meta_config")
p.add_argument('--save_args', type=str, metavar='FILE_PATH',
    help='Save arguments in a command line format to FILE_PATH.')
p.add_argument('--save_json', type=str, metavar='FILE_PATH',
    help='Save arguments in json format to FILE_PATH..')
p.add_argument('--load_json', type=str, metavar='FILE_PATH',
    help='Load arguments from FILE_PATH, in json format. Command line options override json values.')

class Config(dict):
    """This is just a dict that also supports ``.key'' access, making calling config
    values less cumbersome."""
    __slots__ = []

    def __init__(self, *arg, **kw):
        super(Config, self).__init__(*arg, **kw)

    def __getattr__(self, key: str) -> Any:
        if key in self:
            return self[key]
        else:
            return None

def parse(config: Config = None) -> Config:
    """Parse arguments.

    Inputs:
        - config: an existing Config to update.

    Returns:
        A Config with the parsed/updated arguments.
    """
    if config is None:
        config = Config()

    args = parser.parse_args()
    if "load_json" in args and args.load_json:
        # Load json config if there is one.
        with open(args.load_json, 'rt') as f:
            config.update(json.load(f))

    # Parse command line again, with defaults set to the values from the json
    # config. This is needed to allow command line arguments to override the
    # json config.
    parser.set_defaults(**config)
    args = parser.parse_args()
    config.update(vars(args))

    # Optionally save passed arguments
    if config.save_args:
        with open(config.save_args, "w") as f:
            f.write("\n".join(sys.argv[1:]))
        logging.info("Saving arguments to %s.", config.save_args)

    if config.save_json:
        with open(config.save_json, 'wt') as f:
            json.dumps(args, f, indent=4, sort_keys=True)

    return config

