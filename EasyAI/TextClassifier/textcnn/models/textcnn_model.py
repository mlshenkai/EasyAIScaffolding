# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/3/24 1:53 PM
# @File: textcnn_model
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNNModel(nn.Module):
    """Convolutional Neural Networks for Sentence Classification"""

    def __init__(
        self,
        vocab_size,
        num_classes,
        embed_size=200,
        filter_sizes=(2, 3, 4),
        num_filters=256,
        dropout_rate=0.5,
    ):
        """
        Init the TextCNNModel
        @param vocab_size:
        @param num_classes:
        @param embed_size:
        @param filter_sizes: 卷积核尺寸
        @param num_filters: 卷积核数量(channels数)
        @param dropout_rate:
        """
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_size, padding_idx=vocab_size - 1
        )
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed_size)) for k in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
