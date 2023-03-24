# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/3/24 2:05 PM
# @File: textrnn_model
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRNNAttModel(nn.Module):
    """Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification"""

    def __init__(
        self,
        vocab_size,
        num_classes,
        embed_size=200,
        hidden_size=128,
        num_layers=2,
        dropout_rate=0.5,
    ):
        """
        TextRNN-Att model
        @param vocab_size:
        @param num_classes:
        @param embed_size: 字向量维度
        @param hidden_size: lstm隐藏层
        @param num_layers: lstm层数
        @param dropout_rate:
        """
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_size, padding_idx=vocab_size - 1
        )
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate,
        )
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size * 2, int(hidden_size / 2))
        self.fc2 = nn.Linear(int(hidden_size / 2), num_classes)

    def forward(self, x):
        emb = self.embedding(x[0])  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(
            emb
        )  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc2(out)  # [128, 64]
        return out
