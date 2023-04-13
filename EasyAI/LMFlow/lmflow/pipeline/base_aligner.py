# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/13 1:43 PM
# @File: base_aligner
# @Email: mlshenkai@163.com

from .base_pipeline import BasePipeline


class BaseAligner(BasePipeline):
    def __init__(self, *args, **kwargs):
        pass

    def _check_if_aligner(self, model, dataset, reward_model):
        """
        # TODO check model is aligner ,dataset is compatible and reward is aligner
        Args:
            model:
            dataset:
            reward_model:

        Returns:

        """
        pass

    def align(self, model, dataset, reward_model):
        raise NotImplementedError("align is not implement")
