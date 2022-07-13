import torch
import os
from torch.utils.tensorboard import SummaryWriter


def save_pretrained(model, path, filename):
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, filename))


def createtb_log(log_dir):
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    return SummaryWriter(log_dir=log_dir)


def write_tb(tb_obj, tags, data, epoch):
    for t, d in zip(tags, data):
        tb_obj.add_scalar(t, d, epoch)
