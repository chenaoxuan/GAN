"""
Author:  Cax
File:    miscellaneous
Project: GAN
Time:    2022/7/3
Des:     Some fragmentary unsorted functions.
"""
import os
from .distributed import is_main_process


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def save_config(config, save_path):
    if is_main_process():
        with open(save_path, 'w', encoding='GBK') as f:
            f.write(config.__str__())

