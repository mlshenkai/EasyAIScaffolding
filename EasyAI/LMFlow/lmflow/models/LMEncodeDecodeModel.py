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
import torch


class LMEncodeDecodeModel(LMBaseModel, BaseEncodeDecodeModel):
    def check_is_encode_decode_model(self):
        pass

    def generate(self, inputs, *args, **kwargs):
        with torch.no_grad():
            if self.use_gpu:
                outputs = self.ds_engine.module.generate(
                    input_ids=inputs,
                    synced_gpus=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    *args,
                    **kwargs,
                )
            else:
                outputs = self.backend_model.generator(
                    input_ids=inputs,
                    synced_gpus=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    *args,
                    **kwargs,
                )
        return outputs

