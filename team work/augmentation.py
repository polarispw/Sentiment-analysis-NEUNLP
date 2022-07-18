from transformers import pipeline
import random
from tqdm import tqdm


source_file = 'data/t4.tsv'
aug_file = 'data/aug_train.tsv'
target_file = 'data/train_c1.tsv'


def aug_by_bert():
    unmasker = pipeline("fill-mask", model="distilbert-base-uncased")
    cnt = 0
    corpus = []
    aug_corpus = []
    with open(source_file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            cnt += 1
            c, sentence = line.strip().split(",", 1)
            words = sentence.strip().split()
            corpus.append((c, words))
            if (cnt % 20000 == 0):
                for it in tqdm(corpus):
                    # cnt += 1
                    cls, words = it
                    idx = random.randint(0, len(words) - 1)
                    word_list = [w for w in words]
                    word_list[idx] = "[MASK]"
                    # print("Masked Sentence: " + " ".join(new_sent))
                    predict = unmasker(" ".join(word_list))
                    # print("\n".join([str(i) for i in predict]))
                    predict = [i for i in predict if i["token_str"] != words[idx].lower()]
                    # print(predict)
                    score = 0
                    for res in predict:
                        if res['score'] > score:
                            word_list[idx] = res['token_str']
                            score = res['score']
                    sen1 = cls + ", " + " ".join(words)
                    sen2 = cls + ", " + " ".join(word_list)
                    aug_corpus.append(sen1)
                    aug_corpus.append(sen2)
                with open(aug_file, 'a', encoding='utf-8') as ff:
                    ff.write('\n')
                    for sen in aug_corpus:
                        ff.write(str(sen) + '\n')
                aug_corpus.clear()
                corpus.clear()
                print("write done!")
                print(aug_corpus)


def change_label(origin, target):
    with open(source_file, 'r', encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            cls, sentence = line.strip().split(",", 1)
            cls = target if cls in origin else cls
            with open(target_file, 'a', encoding='utf-8') as ff:
                ff.write(cls + ', ' + sentence)
