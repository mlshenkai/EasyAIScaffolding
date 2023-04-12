# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/3/26 7:43 PM
# @File: base_model
# @Email: mlshenkai@163.com
class EasyBaseModel:
    """
    Abstract class for classifier
    """

    def train(self, *args, **kwargs):
        raise NotImplementedError("train method not implemented.")

    def predict(self, *args, **kwargs):
        raise NotImplementedError("predict method not implemented.")

    def evaluate_model(self, **kwargs):
        raise NotImplementedError("evaluate_model method not implemented.")

    def evaluate(self, **kwargs):
        raise NotImplementedError("evaluate method not implemented.")

    def load_model(self):
        raise NotImplementedError("load method not implemented.")

    def save_model(self):
        raise NotImplementedError("save method not implemented.")
