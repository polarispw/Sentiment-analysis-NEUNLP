from gensim.models.word2vec import PathLineSentences
from gensim.models import word2vec
import logging


# 加载分词后的文本
sentences = PathLineSentences("data")
# 训练模型
model = word2vec.Word2Vec(sentences, size=256, hs=1, min_count=1, window=3)
model.wv.save_word2vec_format("penn_256dim_word_vector.bin", binary=True)