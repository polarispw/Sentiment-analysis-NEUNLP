import torch
import torch.nn
from model import AttRNN
from mydataset import create_vocab


device = "cpu"
model_path = "best_model.pth"
sentence = "<sos> I like to sleep in the summer afternoon and"
words = sentence.split(" ")

w2i, i2w = create_vocab("penn/train.txt")
model_dict = torch.load(model_path, map_location=device)
model = AttRNN(len(w2i), device=device)
model.load_state_dict(model_dict)
model.eval()

for i in range(10):
    sen2idx = []
    for word in words[-10:]:
        try:
            sen2idx += [w2i[word]]
        except:
            sen2idx += [w2i['<unk>']]
    sen2idx = torch.tensor(sen2idx).to(device)
    sen2idx = sen2idx.unsqueeze(0)
    pre = model(sen2idx)
    next_word = pre.data.max(1, keepdim=True)[1][0][0].item()
    try:
        words += [i2w[next_word]]
    except:
        words += ["<unk>"]

for word in words:
    print(word, end=" ")
