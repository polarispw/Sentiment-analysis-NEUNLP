import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import logging
from models import Bert_FFN
from dataset import SCDataset


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logging.set_verbosity_error()


def coffate_fn_test(examples):
    inputs, targets = [], []
    for sent in examples:
        inputs.append(sent)
        targets.append(-1)
    inputs = tokenizer(inputs,
                       padding=True,
                       truncation=True,
                       return_tensors="pt",
                       max_length=512)
    targets = torch.tensor(targets)
    return inputs, targets


if __name__ == '__main__':
    test_data_path = "./data/test.tsv"
    test_data = []
    with open(test_data_path, 'r', encoding="utf-8") as fr:
        for line in fr.readlines():
            sentence = line.strip()
            test_data.append(sentence)
    test_dataset = SCDataset(test_data)
    test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             collate_fn=coffate_fn_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model_name = 'bert-base-uncased'
    model_path = 'bert_sc_07_12_10_55/checkpoints-4/model.pth'
    model = torch.load(model_path)
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    res = []
    for batch in tqdm(test_dataloader, desc=f"Testing"):
        inputs, targets = [x.to(device) for x in batch]
        with torch.no_grad():
            bert_output = model(inputs)
            #print(bert_output.argmax(dim=1).data.item())
            res.append(str(bert_output.argmax(dim=1).data.item()))

    with open(model_path + '.csv', 'w', encoding='utf-8') as fw:
        for item in res:
            fw.write(item+'\n')