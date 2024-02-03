import os
import sys
import torch
import torch.nn as nn
import configargparse

def create_directory(path: str):
    """
    create directory if path does not exists
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print("\033[0;31;40m[Create Directory]\033[0m{}".format(path))
    return path

def log_metrics(path: str, metrics: dict):
    if path == '':
        return
    with open(path, 'w+') as f:
        for k, v in metrics.items():
            f.write('{}={}\n'.format(k, v))

def load_metrics(path: str):
    if path == '':
        return
    with open(path, 'r') as f:
        metrics = {}
        for line in f.readlines():
            kv = line.split('=')
            if kv[0] in ['fp', 'fn', 'size', 'num', 'tp']:
                metrics[kv[0]] = int(kv[1])
            elif kv[0] == 'architecture':
                metrics[kv[0]] = kv[1]
            else:
                metrics[kv[0]] = float(kv[1])
        
    return metrics
        
            

def print_dict(d: dict):
    for k, v in d.items():
        print('{}: {}'.format(k, v))

def log_args(path: str, args):
    if path == '':
        return
    log_metrics(path, args.__dict__)

def log_net(path: str, model: nn.Module):
    with open(path, 'w+') as f:
        f.write(model)

