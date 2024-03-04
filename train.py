import os
import argparse
import torch
from utils.Config import Config
from trainer import NetworkManager


def main():
    options = {
        'epochs': 60,
        'batch_size': 16,
        'base_lr': 0.001,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'img_size': 448,
        'device': torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')
    }

    path = {
        'data': Config.data_path,
        'model_save': Config.model_save_path
    }

    for p in path:
        print(p)
        print(path[p])
        assert os.path.isdir(path[p])

    manager = NetworkManager(options, path)
    manager.train()

if __name__ == '__main__':
    main()
