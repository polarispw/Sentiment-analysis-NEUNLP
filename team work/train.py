import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import time
import argparse
from transformers import BertTokenizer
from transformers.optimization import AdamW
from transformers import logging

import models
from dataset import data_process, SCDataset
from utils import save_pretrained, createtb_log, write_tb
from optim import collate_para, build_scheduler


def collate_fn(examples):
    inputs, targets = [], []
    for polar, sent in examples:
        inputs.append(sent)
        targets.append(int(polar))
    inputs = tokenizer(inputs,
                       padding=True,
                       truncation=True,
                       return_tensors="pt",
                       max_length=512)
    targets = torch.tensor(targets)
    return inputs, targets


def train(args):

    data_path = args.data_path
    batch_size = args.batch_size

    train_data, valid_data, categories, exceeding_ratio = data_process(
        train_data_path=data_path + 'train.tsv',
        valid_data_path=data_path + 'valid.tsv')

    train_dataset = SCDataset(train_data)
    valid_dataset = SCDataset(valid_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, collate_fn=collate_fn)

    model = models.Bert_LSTM(len(categories), pretrained_model_name)
    model.to(device)

    para_groups = collate_para(model, [args.lr], args.lr_decay_rate, discr=args.discriminative_lr)
    optimizer = AdamW(para_groups, lr=args.lr)
    total_steps = len(train_data) * args.epochs
    scheduler = build_scheduler(args.scheduler,
                                optimizer,
                                int(total_steps/10), # 10%的steps用来warm up
                                total_steps)

    CE_loss = nn.CrossEntropyLoss()

    timestamp = time.strftime("%m%d_%H-%M", time.localtime())
    log_dir = os.path.join("./runs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    arg_dict = vars(args)
    with open(log_dir + "/arg_list.txt", "w") as f:
        for k, v in arg_dict.items():
            f.write(f"{k} : {v}\n")
    tb_log = createtb_log(log_dir)

    total_step = 0
    for epoch in range(args.epochs):

        total_loss = 0
        train_dataloader = tqdm(train_dataloader, file=sys.stdout)
        model.train()
        for step, batch in enumerate(train_dataloader):
            inputs, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            bert_output = model(inputs)

            loss = CE_loss(bert_output, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            train_dataloader.desc = f"Training Epoch {epoch} loss: {total_loss / (step + 1):.6f},"
            if total_step % 250 == 0:
                write_tb(tb_log, ['train loss'], [total_loss / (step + 1)], total_step)
            total_step += 1

        acc = 0
        valid_dataloader = tqdm(valid_dataloader, file=sys.stdout)
        model.eval()
        for step, batch in enumerate(valid_dataloader):
            inputs, targets = [x.to(device) for x in batch]
            with torch.no_grad():
                bert_output = model(inputs)
                acc += (bert_output.argmax(dim=1) == targets).sum().item()
            valid_dataloader.desc = f"Valid_Acc: {acc / (step+1):.6f}"

        write_tb(tb_log, ['valid acc'], [acc / len(valid_dataloader)], epoch)

        if epoch % 1 == 0:
            filename = f"bert_sc_epoch {epoch}.pth"
            save_pretrained(model, log_dir, filename)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    logging.set_verbosity_error()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--data-path', type=str, default="./data/")

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--discriminative_lr', type=bool, default=False)
    parser.add_argument('--lr-decay-rate', type=float, default=0.9)
    parser.add_argument('--scheduler', type=str, default='linear')

    parser.add_argument('--model', type=str, default=pretrained_model_name)
    parser.add_argument('--n-step', type=int, default=5)
    parser.add_argument('--embedding-size', type=int, default=512)
    parser.add_argument('--n-hidden', type=int, default=256)

    parser.add_argument('--save-step', type=int, default=500)

    opt = parser.parse_args()

    train(opt)
