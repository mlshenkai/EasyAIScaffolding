# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/14 7:38 PM
# @File: base_decode_only_model
# @Email: mlshenkai@163.com
from abc import ABC


class BaseDecodeOnlyModel(ABC):
    def check_is_base_only_model(self):
        # TODO (@Watcher) add check func
        raise NotImplementedError(
            "this func should be implement, because you must insure model is decode only"
        )
