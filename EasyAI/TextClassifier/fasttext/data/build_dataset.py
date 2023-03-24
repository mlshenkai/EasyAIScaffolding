# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/3/24 1:42 PM
# @File: build_dataset
# @Email: mlshenkai@163.com
from EasyAI.TextClassifier.data_helper import load_vocab, build_vocab
from loguru import logger
import os
import json
import torch


def build_dataset(
    tokenizer,
    X,
    y,
    word_vocab_path,
    label_vocab_path,
    max_vocab_size=10000,
    max_seq_length=128,
    unk_token="[UNK]",
    pad_token="[PAD]",
    n_gram_vocab=250499,
    enable_ngram=True,
):
    if os.path.exists(word_vocab_path):
        word_id_map = load_vocab(word_vocab_path)
    else:
        word_id_map = build_vocab(
            X,
            tokenizer=tokenizer,
            max_size=max_vocab_size,
            min_freq=1,
            unk_token=unk_token,
            pad_token=pad_token,
        )
        json.dump(
            word_id_map,
            open(word_vocab_path, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=4,
        )
        logger.debug("save vocab_path: {}".format(word_vocab_path))
    logger.debug(
        f"word vocab size: {len(word_id_map)}, word_vocab_path: {word_vocab_path}"
    )

    if os.path.exists(label_vocab_path):
        label_id_map = load_vocab(label_vocab_path)
    else:
        id_label_map = {id: v for id, v in enumerate(set(y.tolist()))}
        label_id_map = {v: k for k, v in id_label_map.items()}
        json.dump(
            label_id_map,
            open(label_vocab_path, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=4,
        )
        logger.debug("save label_vocab_path: {}".format(label_vocab_path))
    logger.debug(
        f"label vocab size: {len(label_id_map)}, label_vocab_path: {label_vocab_path}"
    )

    def biGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    def load_dataset(X, y, max_seq_length=128):
        contents = []
        for content, label in zip(X, y):
            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if max_seq_length:
                if len(token) < max_seq_length:
                    token.extend([pad_token] * (max_seq_length - len(token)))
                else:
                    token = token[:max_seq_length]
                    seq_len = max_seq_length
            # word to id
            for word in token:
                words_line.append(word_id_map.get(word, word_id_map.get(unk_token)))
            label_id = label_id_map.get(label)
            # fasttext ngram
            bigram = []
            trigram = []
            if enable_ngram:
                buckets = n_gram_vocab
                # ------ngram------
                for i in range(max_seq_length):
                    bigram.append(biGramHash(words_line, i, buckets))
                    trigram.append(triGramHash(words_line, i, buckets))
                # -----------------
            else:
                bigram = [0] * max_seq_length
                trigram = [0] * max_seq_length
            contents.append((words_line, label_id, seq_len, bigram, trigram))
        return contents

    dataset = load_dataset(X, y, max_seq_length)
    return dataset, word_id_map, label_id_map


class DatasetIterater:
    def __init__(self, dataset, device, batch_size=32):
        self.batch_size = batch_size
        self.dataset = dataset
        self.n_batches = len(dataset) // batch_size if len(dataset) > batch_size else 1
        self.residue = False  # 记录batch数量是否为整数
        if len(dataset) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        # pad_token前的长度(超过max_seq_length的设为max_seq_length)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, bigram, trigram), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size : len(self.dataset)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[
                self.index * self.batch_size : (self.index + 1) * self.batch_size
            ]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, device, batch_size=32):
    return DatasetIterater(dataset, device, batch_size)
