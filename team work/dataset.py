from torch.utils.data import Dataset


def data_process(train_data_path, valid_data_path):
    train_data = []
    valid_data = []
    categories = set()
    exceeding_sample = 0
    with open(train_data_path, 'r', encoding="utf-8") as fr:
        for line in fr.readlines():
            cls, sentence = line.strip().split(",", 1)
            exceeding_sample += 1 if len(sentence) > 512 else 0
            train_data.append((cls, sentence))
            categories.add(cls)

    with open(valid_data_path, 'r', encoding="utf-8") as fr1:
        for line in fr1.readlines():
            cls, sentence = line.strip().split(",", 1)
            valid_data.append((cls, sentence))

    exceeding_ratio = 100 * (exceeding_sample / len(train_data))
    return train_data, valid_data, categories, exceeding_ratio


class SCDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.dataset[index]
