import torch
import torch.nn as nn
from torch.nn.functional import softmax
from utils import load_word_emb_pretrained


# Model
class FFNLM(nn.Module):
    def __init__(self, n_step, n_class, m=512, n_hidden=256):
        super(FFNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)
        self.w1 = nn.Linear(n_step * m, n_hidden, bias=False)
        self.b1 = nn.Parameter(torch.ones(n_hidden))
        self.w2 = nn.Linear(n_hidden, n_class, bias=False)
        self.w3 = nn.Linear(n_step * m, n_class, bias=False)
        self.n_step = n_step
        self.m = m

    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, self.n_step * self.m)    # X
        Y1 = torch.tanh(self.b1 + self.w1(X)) #Y1 b1 w1
        b2 = self.w3(X)  #b2  W2
        Y2 = b2 + self.w2(Y1) #Y2
        return Y2


class TextRNN(nn.Module):
    def __init__(self, n_class, emb_size=512, n_hidden=256, word_emb=''):
        super(TextRNN, self).__init__()
        if word_emb != '':
            self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        else:
            word_vectors = load_word_emb_pretrained(word_emb)
            nn.Embedding.from_pretrained(word_vectors)
            self.C.weight.requires_grad = False
        self.rnn = nn.RNN(input_size=emb_size, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1) # X : [n_step, batch_size, embeding size]
        outputs, hidden = self.rnn(X)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b # model : [batch_size, n_class]
        return model


class AttRNN(nn.Module):
    def __init__(self, n_class, emb_size=256, n_hidden=512, device='cuda:0'):
        super(AttRNN, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        self.rnn = nn.RNN(input_size=emb_size, hidden_size=n_hidden)
        self.W = nn.Linear(2*n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))
        self.n_hidden = n_hidden
        self.device = device

    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1)  # X : [n_step, batch_size, embeding size]
        outputs, hidden = self.rnn(X)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        output = outputs[-1]
        attention = []
        for it in outputs[:-1]:
            attention.append(torch.mul(it, output).sum(dim=1).tolist())
        self.attention = torch.tensor(attention).to(self.device)
        self.attention = self.attention.transpose(0, 1)
        self.attention = softmax(self.attention, dim=1).transpose(0, 1)
        # get soft attention
        attention_output = torch.zeros(outputs.size()[1], self.n_hidden).to(self.device)
        for i in range(outputs.size()[0] - 1):
            attention_output += torch.mul(self.attention[i], outputs[i].transpose(0, 1)).transpose(0, 1)
        output = torch.cat((attention_output, output), 1)
        # joint ouput output:[batch_size, 2*n_hidden]
        model = self.W(output) + self.b  # model : [batch_size, n_class]
        return model


class TextLSTM(nn.Module):
    def __init__(self, n_class, emb_size=512, n_hidden=256, device='cuda:0'):
        super(TextLSTM, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))
        self.n_hidden = n_hidden
        self.device = device

    def forward(self, X):
        X = self.C(X)
        input = X.transpose(0, 1)  # X : [n_step, batch_size, n_class]

        hidden_state = torch.zeros(1, len(X), self.n_hidden).to(self.device)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state = torch.zeros(1, len(X), self.n_hidden).to(self.device)     # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model


class BiLSTM(nn.Module):
    def __init__(self, n_class, emb_size=512, n_hidden=256, device='cuda:0'):
        super(BiLSTM, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=n_hidden, bidirectional=True)
        self.W = nn.Linear(n_hidden * 2, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))
        self.n_hidden = n_hidden
        self.device = device

    def forward(self, X):
        X = self.C(X)
        input = X.transpose(0, 1)  # input : [n_step, batch_size, n_class]

        hidden_state = torch.zeros(1*2, len(X), self.n_hidden).to(self.device)   # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.zeros(1*2, len(X), self.n_hidden).to(self.device)     # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model
