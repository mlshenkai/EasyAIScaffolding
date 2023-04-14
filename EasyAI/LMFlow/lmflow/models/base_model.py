# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/14 3:03 PM
# @File: base_model
# @Email: mlshenkai@163.com
from abc import ABC
from ..args import ModelArguments
from ..data.dataset import LMDataset
from typing import Union, List


class LMBaseModel(ABC):
    def __init__(
        self,
        model_args: ModelArguments,
        tune_strategy="none",
        ds_config=None,
        use_gpu: bool = True,
        *args,
        **kwargs,
    ):
        self.model_args = model_args
        self.tune_strategy = tune_strategy
        self.ds_config = ds_config
        self.use_gpu = use_gpu
        self.args = args
        self.kwargs = kwargs

    def tokenize(self, dataset: LMDataset, *args, **kwargs):
        raise NotImplementedError("tokenize func must be implement")

    def encode(self, inputs: Union[str, List[str]], *args, **kwargs):
        raise NotImplementedError("encode func must be implement")

    def decode(self, outputs, *args, **kwargs) -> Union[str, List[str]]:
        raise NotImplementedError("decode func must be implement")

    def inference(self, inputs, *args, **kwargs):
        raise NotImplementedError("inference func must be implement")

    def merge_tune_weight(self, *args, **kwargs):
        """
        if you have tune weight to merge base model you must implement this func
        Returns:

        """
        pass

    def save(self, dir: str, save_full_name=False, *args, **kwargs):
        """
        if you have tune weight to save you must implement this func
        Returns:

        """
        pass

    def get_max_length(self):
        raise NotImplementedError("get_max_length func must be implement")

    def get_tokenizer(self):
        raise NotImplementedError("get_tokenizer func must be implement")

    def get_backend_model(self):
        raise NotImplementedError("get_backend_model func must be implement")
