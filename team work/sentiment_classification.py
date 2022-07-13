import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import time
from transformers import BertTokenizer
from transformers import logging

from dataset import data_process, SCDataset
import models
from utils import save_pretrained, createtb_log, write_tb


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


def train():

    train_data, valid_data, categories, exceeding_ratio= data_process(train_data_path=data_path + 'train.tsv', valid_data_path=data_path + 'valid.tsv')

    train_dataset = SCDataset(train_data)
    valid_dataset = SCDataset(valid_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, collate_fn=collate_fn)

    model = models.Bert_LSTM(len(categories), pretrained_model_name)
    model.to(device)

    optimizer = Adam(model.parameters(), learning_rate)
    CE_loss = nn.CrossEntropyLoss()

    timestamp = time.strftime("%m%d_%H:%M", time.localtime())
    log_dir = os.path.join("runs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    tb_log = createtb_log(log_dir)

    model.train()
    for epoch in range(1, num_epoch + 1):
        total_loss = 0
        train_dataloader = tqdm(train_dataloader, file=sys.stdout)
        for step, batch in enumerate(train_dataloader):
            inputs, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            bert_output = model(inputs)
            loss = CE_loss(bert_output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            train_dataloader.desc = f"Training Epoch {epoch} loss: {total_loss / (step + 1):.6f},"

        acc = 0
        valid_dataloader = tqdm(valid_dataloader, file=sys.stdout)
        for step, batch in enumerate(valid_dataloader):
            inputs, targets = [x.to(device) for x in batch]
            with torch.no_grad():
                bert_output = model(inputs)
                acc += (bert_output.argmax(dim=1) == targets).sum().item()
            valid_dataloader.desc = f"Valid_Acc: {acc / (step+1):.6f}"

        write_tb(tb_log,
                 ['train loss', 'valid acc'],
                 [total_loss / (step + 1), acc / len(valid_dataloader)],
                 epoch)

        if epoch % check_step == 0:
            filename = f"bert_sc_epoch {epoch}.pth"
            save_pretrained(model, log_dir, filename)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    logging.set_verbosity_error()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_model_name = 'bert-base-uncased'

    batch_size = 16
    num_epoch = 5
    check_step = 1
    data_path = "./data/"
    learning_rate = 1e-5

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    train()

