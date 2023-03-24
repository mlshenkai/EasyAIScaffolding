# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/3/24 1:45 PM
# @File: fasttext_model
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastTextModel(nn.Module):
    """Bag of Tricks for Efficient Text Classification"""

    def __init__(
        self,
        vocab_size,
        num_classes,
        embed_size=200,
        n_gram_vocab=250499,
        hidden_size=256,
        dropout_rate=0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_size, padding_idx=vocab_size - 1
        )
        self.embedding_ngram2 = nn.Embedding(n_gram_vocab, embed_size)
        self.embedding_ngram3 = nn.Embedding(n_gram_vocab, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embed_size * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
