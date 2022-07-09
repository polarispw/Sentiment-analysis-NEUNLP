import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import os
import sys
import json
import math
import argparse
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from mydataset import create_vocab, load_corpus, PennDataset
from model import FFNLM, TextRNN, TextLSTM, BiLSTM
from utils import train_one_epoch, evaluate, get_parameter_number


def train(args):

    train_path = os.path.join(args.data_path, "train.txt")
    valid_path = os.path.join(args.data_path, "valid.txt")
    device = args.device
    epochs = args.epochs
    batch_size = args.batch_size
    n_step = args.n_step # number of steps, n-1 in paper
    m = args.embedding_size # embedding size, m in paper
    n_hidden = args.n_hidden

    # prepare data
    word2idx, idx2word = create_vocab(train_path)
    train_corpus = load_corpus(train_path)
    valid_corpus = load_corpus(valid_path)

    train_dataset = PennDataset(train_corpus, word2idx, n_step=n_step)

    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=train_dataset.collate_fn,
            num_workers=0,
            shuffle=True
        )

    valid_dataset = PennDataset(valid_corpus, word2idx, n_step=n_step)

    valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            collate_fn=train_dataset.collate_fn,
            num_workers=0,
            shuffle=True
        )

    # get modules ready
    if args.model == "ffn":
        model = FFNLM(n_step, len(word2idx), m=m, n_hidden=n_hidden)
    elif args.model == "rnn":
        model = TextRNN(len(word2idx), emb_size=m, n_hidden=n_hidden, word_emb=args.word_emb_path)
    elif args.model == "lstm":
        model = TextLSTM(len(word2idx), emb_size=m, n_hidden=n_hidden, device=device)
    elif args.model == "bilstm":
        model = BiLSTM(len(word2idx), emb_size=m, n_hidden=n_hidden, device=device)
    model.to(device)

    # if args.word_emb_path is not None:
    #     for name, para in model.named_parameters():
    #         if layers_to_train not in name:
    #             para.requires_grad_(False)
    #         else:
    #             break
    pg = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer == 'SGD':
        # SGD
        optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    elif args.optimizer == 'RMSprop':
        # RMSprop
        optimizer = optim.RMSprop(pg, lr=args.lr, weight_decay=1E-4)
    else:
        # Adam
        optimizer = optim.Adam(pg, lr=args.lr, weight_decay=1E-6)

    def lr_lambda(current_step: int):
        num_warmup_steps = 5 if "warmup" in args.scheduler else -1
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if "cosine" in args.scheduler:
            return ((1 + math.cos(current_step * math.pi / args.epochs)) / 2) * (1 - 1E-3) + 1E-3
        elif "steps" in args.scheduler:
            lrf_step = [1, 0.2, 0.02, 0.002, 0.0005]
            return lrf_step[int(current_step / 5)]
        return max(1E-4, pow(0.88, int((current_step-1)/1)))

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    print(model)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    if os.path.exists("./runs") is False:
        os.makedirs("./runs")
    log_path = "./runs/{}".format(datetime.datetime.now().strftime("%Y_%m%d-%H_%M_%S"))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    with open(log_path + "/arg_list_epoch[{}].json".format(0), "w") as f:
        f.write(json.dumps(vars(args)))
        f.write("\n**********\n")
        f.write(json.dumps(get_parameter_number(model)))
    tb_writer = SummaryWriter(log_dir=log_path)

    # Training
    for epoch in range(epochs):
        best_val_ppl = 0

        train_loss, train_ppl = train_one_epoch(
                                    model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    clip_grad=args.clip_grad,
                                    epoch=epoch
                                )

        valid_loss, valid_ppl = evaluate(
                                model=model,
                                data_loader=valid_loader,
                                device=device,
                                epoch=epoch
                            )

        scheduler.step()

        tags = ["train_loss", "train_ppl", "val_loss", "val_ppl", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_ppl, epoch)
        tb_writer.add_scalar(tags[2], valid_loss, epoch)
        tb_writer.add_scalar(tags[3], valid_ppl, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if valid_ppl > best_val_ppl:
            best_val_ppl = valid_ppl
            torch.save(model.state_dict(), log_path + "/best_model.pth")


def test(args):
    model = torch.load(args.test_model_path, map_location=args.device)  # load the selected model

    # load the test data
    train_path = os.path.join(args.data_path, "train.txt")
    test_path = os.path.join(args.data_path, "test.txt")
    word2idx, idx2word = create_vocab(train_path)
    test_corpus = load_corpus(test_path)
    dataset = PennDataset(test_corpus, word2idx, n_step=args.n_step)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=0,
        shuffle=True
    )
    model.eval()
    loss_fun = nn.CrossEntropyLoss()
    data_loader = tqdm(dataloader, file=sys.stdout)

    total_loss = 0

    for step, batch in enumerate(data_loader):
        inputs, targets = batch
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        targets = targets.view(-1)
        # output : [batch_size, n_class], target_batch : [batch_size]
        output = model(inputs)

        loss = loss_fun(output, targets)
        total_loss += loss.detach()

    print(f"\n\nTest {0} samples with {args.test_model_path}……………………")
    print('loss =', '{:.6f}'.format(total_loss / (step + 1)),
          'ppl =', '{:.6}'.format(math.exp(total_loss / (step + 1))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data-path', type=str, default='data')

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1E-3)
    parser.add_argument('--optimizer', type=str, default='Adam', help='choose from SGD and Adam')
    parser.add_argument('--scheduler', type=str, default='', help='write your lr schedule keywords')
    parser.add_argument('--clip-grad', type=bool, default=False)
    parser.add_argument('--word-emb-path', type=str, default='')

    parser.add_argument('--model', type=str, default='rnn')
    parser.add_argument('--n-step', type=int, default=5)
    parser.add_argument('--embedding-size', type=int, default=512)
    parser.add_argument('--n-hidden', type=int, default=256)

    parser.add_argument('--test_model_path', type=str, default="best.pth")
    opt = parser.parse_args()

    train(opt)
    # test(opt)