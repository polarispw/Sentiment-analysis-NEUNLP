import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import math
import gensim


def train_one_epoch(model, optimizer, data_loader, device, clip_grad, epoch):

    model.train()
    loss_fun = nn.CrossEntropyLoss()
    data_loader = tqdm(data_loader, file=sys.stdout)

    total_loss = 0
    max_norm = 10

    optimizer.zero_grad()
    for step, batch in enumerate(data_loader):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        targets = targets.view(-1)
        # output : [batch_size, n_class], target_batch : [batch_size]
        output = model(inputs)

        loss = loss_fun(output, targets)
        loss.backward()

        total_loss += loss.detach()
        data_loader.desc = f"[Train epoch {epoch}] loss: {total_loss/(step+1):.6f}, ppl: {math.exp(total_loss/(step+1)):.6}"

        if clip_grad:
            total_norm = nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm, norm_type=2)
            if (step+1) % 201 == 0:
                max_norm = total_norm.item()
                print(max_norm)

        optimizer.step()
        optimizer.zero_grad()

    return total_loss/(step+1), math.exp(total_loss/(step+1))


def evaluate(model, data_loader, device, epoch):

    model.eval()
    loss_fun = nn.CrossEntropyLoss()
    data_loader = tqdm(data_loader, file=sys.stdout)

    total_loss = 0

    for step, batch in enumerate(data_loader):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        targets = targets.view(-1)
        # output : [batch_size, n_class], target_batch : [batch_size]
        output = model(inputs)

        loss = loss_fun(output, targets)
        total_loss += loss.detach()
        data_loader.desc = f"[Valid epoch {epoch}] loss: {total_loss / (step + 1):.6f}, ppl: {math.exp(total_loss / (step + 1)):.6}"

    return total_loss / (step + 1), math.exp(total_loss / (step + 1))


def get_parameter_number(model):
    total_num = 0
    for name, para in model.named_parameters():
        if 'Embedding' not in name:
            total_num += sum(p.numel() for p in para)
    return {'Total parameter': total_num/1000000}


def load_word_emb_pretrained(data_path, voc_size, i2w):
    w2v = gensim.models.KeyedVectors.load_word2vec_format(data_path)
    word_vec = torch.randn([voc_size, 300])
    for i in range(0, voc_size):
        word = i2w[i]
        if word in w2v:
            vec = w2v[word]
            word_vec[i,:] = torch.from_numpy(vec)
    return word_vec
