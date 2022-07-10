import torch
import torch.nn as nn

batch_size = 128
valid_path = 'data/valid.txt'
test_psth = 'data/test.txt'

def make_batch(train_path, word2number_dict, n_step):
    def word2number(n):
        try:
            return word2number_dict[n]
        except:
            return 1   #<unk_word>

    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8') #open the file

    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer

        if len(word) <= n_step:   #pad the sentence
            word = ["<pad>"]*(n_step+1-len(word)) + word

        for word_index in range(len(word)-n_step):
            input = [word2number(n) for n in word[word_index:word_index+n_step]]  # create (1~n-1) as input
            target = word2number(word[word_index+n_step])  # create (n) as target, We usually call this 'casual language model'
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch

def give_valid(word2number_dict, n_step):
    all_input_batch, all_target_batch = make_batch(valid_path, word2number_dict, n_step)

    all_input_batch = torch.LongTensor(all_input_batch)  # list to tensor
    all_target_batch = torch.LongTensor(all_target_batch)

    return  all_input_batch, all_target_batch


def give_test(word2number_dict, n_step):
    all_input_batch, all_target_batch = make_batch(test_psth, word2number_dict, n_step)

    all_input_batch = torch.LongTensor(all_input_batch)  # list to tensor
    all_target_batch = torch.LongTensor(all_target_batch)

    return all_input_batch, all_target_batch