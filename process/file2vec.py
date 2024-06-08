import re
import numpy as np
import nltk
from collections import Counter
import torch
import gensim
from gensim.models import Word2Vec
from torch import nn
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import build_vocab_from_iterator
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def file2data(file_path: str) -> list:
    """_summary_

    Args:
        file_path (str): absa file path

    Returns:
        list: every elements contain 2 item —— sentence, targets
    """
    res = []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 3):
        sentence = lines[i].strip()
        targets = lines[i + 1].strip().split(',')

        for target in targets:
            target = target.strip()
            processed_sentence = re.sub(r'\$T\$', target, sentence)

            data_tuple = (processed_sentence, target)
            res.append(data_tuple)
    return res



def tokenize_sentence(sentence):
    return word_tokenize(sentence)

def build_vocab(data_tuples):
    counter = Counter()
    for sentence, _ in data_tuples:
        tokens = tokenize_sentence(sentence)
        counter.update(tokens)
    return build_vocab_from_iterator([counter.keys()], specials=["<unk>", "<pad>"])

def sentence_to_tensor(sentence, vocab):
    tokens = tokenize_sentence(sentence)
    token_ids = [vocab[token] for token in tokens]
    return torch.tensor(token_ids, dtype=torch.long)



def handle(data):
    vocab = build_vocab(data)
    vocab.set_default_index(vocab["<unk>"])
    vocab_list = vocab.get_itos()
    tensor_data = [sentence_to_tensor(sentence, vocab) for sentence, _ in data]
    padded_tensor_data = pad_sequence(tensor_data, batch_first=True, padding_value=vocab["<pad>"])
    word_sentences = [[vocab_list[idx] for idx in indexed_sentence] for indexed_sentence in padded_tensor_data]
    word2vec_model = Word2Vec(sentences=word_sentences, vector_size=100, window=5, min_count=1, sg=1)
    vocab_size = len(vocab_list)
    embedding_dim = word2vec_model.vector_size
    embedding_weights = np.zeros((vocab_size, embedding_dim))
    for idx, word in enumerate(vocab_list):
        if word in word2vec_model.wv:
            embedding_weights[idx] = word2vec_model.wv[word]
        else:
            embedding_weights[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
            # 将权重矩阵转换为 PyTorch 张量
    embedding_weights = torch.tensor(embedding_weights, dtype=torch.float32)

    # 创建 Embedding 层并初始化权重
    embedding = nn.Embedding(vocab_size, embedding_dim)
    embedding.weight = nn.Parameter(embedding_weights)

    return vocab, vocab_list, padded_tensor_data, embedding


def trans2matrix(idx_data, embedding):
    indices = []
    for i in idx_data:
        indices.append(i)
    return embedding(torch.tensor(indices))



