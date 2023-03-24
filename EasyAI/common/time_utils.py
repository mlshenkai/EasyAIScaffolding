# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/3/24 11:57 AM
# @File: time_utils
# @Email: mlshenkai@163.com

import time
from datetime import timedelta


def get_time_spend(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def init_network(model, method="xavier", exclude="embedding"):
    """权重初始化，默认xavier"""
    import torch.nn as nn

    for name, w in model.named_parameters():
        if exclude not in name:
            if "weight" in name:
                if method == "xavier":
                    nn.init.xavier_normal_(w)
                elif method == "kaiming":
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            if "bias" in name:
                nn.init.constant_(w, 0)
