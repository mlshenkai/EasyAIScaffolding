# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/14 7:42 PM
# @File: base_encode_decode_model
# @Email: mlshenkai@163.com
from abc import ABC


class BaseEncodeDecodeModel(ABC):
    def check_is_encode_decode_model(self):
        # TODO (@Watcher) add check func
        raise NotImplementedError(
            "this func should be implement, because you must insure model is encode-decode model"
        )
