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

    def __init__(
        self,
        model_args: ModelArguments,
        tune_strategy="none",
        ds_config=None,
        use_gpu: bool = True,
        *args,
        **kwargs,
    ):
        super(LMEncodeDecodeModel, self).__init__(
            model_args, tune_strategy, ds_config, use_gpu, *args, **kwargs
        )
        pass

    def tokenize(self, dataset: LMDataset, *args, **kwargs):
        pass

    def encode(self, inputs: Union[str, List[str]], *args, **kwargs):
        pass

    def decode(self, outputs, *args, **kwargs) -> Union[str, List[str]]:
        pass

    def inference(self, inputs, *args, **kwargs):
        pass

    def get_max_length(self):
        pass

    def get_tokenizer(self):
        pass

    def get_backend_model(self):
        pass
