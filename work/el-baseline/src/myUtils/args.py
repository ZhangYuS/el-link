import argparse
import json
import os
from os.path import join
from loguru import logger


def get_parser():
    parser = argparse.ArgumentParser()

    ## path manager
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output")

    ## training args
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')

    ## batch size and device
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=128)

    ## device
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_cuda", action='store_true')

    ## optimizer
    parser.add_argument("--lr", "--learning_rate", type=float, default=5e-5, dest="learning_rate")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_decay_rate", type=float, default=0.2)
    parser.add_argument("--lr_decay_steps", type=float, default=5000)

    ## iteration
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=200000)

    ## logging
    parser.add_argument("--logging_steps", type=int, default=0)

    ## save
    parser.add_argument("--save_steps", type=int, default=500)

    ## develop and debug
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--load_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1980)

    ## task specified
    parser.add_argument("--select_methods", type=str, default='mean')
    parser.add_argument("--eval_num", type=int, default=500)

    ## predict
    parser.add_argument("--mode", type=str, default=None)

    parser.add_argument("--other_mention", action='store_true')

    return parser.parse_args()


class VersionConfig:
    def __init__(self,
                 sim_loss_func=None,
                 max_seq_length=256
                 ):
        self.sim_loss_func = sim_loss_func
        self.max_seq_length = max_seq_length

    def load(self, cfg_dir):
        cfg_path = join(cfg_dir, 'version_config.json')
        if not os.path.exists(cfg_path):
            logger.warning("there is no version_config file, make sure being loading old version!")
        else:
            params = json.load(open(cfg_path, encoding='utf8'))
            self.__init__(**params)

    def dump(self, cfg_dir):
        params = {}
        for var in self.__dict__:
            if not var.startswith('_'):
                params[var] = getattr(self, var)
        json.dump(params, open(join(cfg_dir, 'version_config.json'), 'w', encoding='utf8'), ensure_ascii=False)

