import torch
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
from collections import Counter, OrderedDict,defaultdict
from tqdm import tqdm


class PennDataset(Dataset):
    def __init__(self, corpus, vocab, n_step=5):
        self.data = []
        self.vocab = vocab
        for sen in tqdm(corpus, desc="Dataset Construction"):
            if len(sen) <= n_step:  # pad the sentence
                sen = ["<pad>"] * (n_step + 1 - len(sen)) + sen
            for i in range(len(sen)-n_step):
                inputs = []
                for word in sen[i: i + n_step]:
                    inputs += [self.word2idx(word)]
                target = [self.word2idx(word)]
                self.data.append([inputs, target])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, phrase):
        inputs = torch.tensor([item[0] for item in phrase])
        targets = torch.tensor([item[1] for item in phrase])
        return (inputs, targets)

    def word2idx(self, str):
        try:
            return self.vocab[str]
        except:
            return self.vocab['<unk>']


class OrderedCounter(Counter, OrderedDict):  #将输入的句子里的新单词添加到字典中，并记录该单词的出现次数
    """Counter that remembers the order elements are first encountered"""
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def create_vocab(path):
    with open(path, 'r') as text:
        tokenizer = TweetTokenizer(preserve_case=False)
        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()
        sentences = []
        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)
        for i, line in enumerate(text):
            words = tokenizer.tokenize(line)
            sentences.append(words)
            w2c.update(words)
        for w, c in w2c.items():
            if c > 0 and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
    return w2i, i2w

def load_corpus(path):
    with open(path, 'r') as text:
        tokenizer = TweetTokenizer(preserve_case=False)
        corpus = []
        for i, line in enumerate(text):
            words = tokenizer.tokenize(line)
            corpus.append(words)
    return corpus
