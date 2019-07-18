import os
import torch
import yaml

DEFAULT_DEVICE = 'cuda:0'


def load_config(config_file):
    assert os.path.exists(config_file)
    config = _load_config_yaml(config_file)

    # Get a device to train on
    device = config.get('device', DEFAULT_DEVICE)
    config['device'] = torch.device(device)
    return config


def _load_config_yaml(config_file):
    return yaml.load(open(config_file, 'r'))
