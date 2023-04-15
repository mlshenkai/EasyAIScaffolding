# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/14 3:29 PM
# @File: LMEncodeDecodeModel
# @Email: mlshenkai@163.com
from typing import Union, List
from ..args import ModelArguments
from .base_model import LMBaseModel
from .base_encode_decode_model import BaseEncodeDecodeModel
from ..data.dataset import LMDataset


class LMEncodeDecodeModel(LMBaseModel, BaseEncodeDecodeModel):
    def check_is_encode_decode_model(self):
        pass

    def inference(self, inputs, *args, **kwargs):
        pass
