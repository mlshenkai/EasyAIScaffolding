# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/14 3:00 PM
# @File: LLMDecodeModel
# @Email: mlshenkai@163.com
import torch
from .base_model import LMBaseModel
from .base_decode_only_model import BaseDecodeOnlyModel



class LMDecodeModel(LMBaseModel, BaseDecodeOnlyModel):
    """
    init model
    """

    def check_is_base_only_model(self):
        pass

    def generate(self, inputs, *args, **kwargs):
        """
        Perform generation process of the model.
        Args:
            inputs:
            *args:
            **kwargs:

        Returns:

        """
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
