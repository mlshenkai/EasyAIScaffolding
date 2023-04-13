# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/12 8:40 PM
# @File: base_tuner
# @Email: mlshenkai@163.com

from .base_pipeline import BasePipeline


class BaseTuner(BasePipeline):
    def __init__(self, *args, **kwargs):
        pass

    def _check_if_tunable(self, model, dataset):
        # TODO check the model is tunable and dataset is compatible
        pass

    def tune(self, model, dataset):
        raise NotImplementedError(".tune is not implement")
