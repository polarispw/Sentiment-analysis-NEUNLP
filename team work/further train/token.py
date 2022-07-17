import tokenizers


tsv_files = ["../data/train.tsv", "../data/valid.tsv"]
corpus = []
for tsv_file in tsv_files:
    with open(tsv_file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            cls, sentence = line.strip().split(",", 1)
            corpus.append(sentence)

with open('corpus.txt', 'w', encoding='utf-8') as f:
    for sen in corpus:
        f.write(sen + '\n')

# create tokenizer
bwpt = tokenizers.BertWordPieceTokenizer()
filepath = "corpus.txt"
bwpt.train(
    files=[filepath],
    vocab_size=50000,
    min_frequency=1,
    limit_alphabet=1000
)
bwpt.save_model('../further pretrain/')
