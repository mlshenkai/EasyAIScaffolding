# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/12 3:52 PM
# @File: auto_model
# @Email: mlshenkai@163.com
from .LMDecodeModel import LMDecodeModel
from .LMEncodeDecodeModel import LMEncodeDecodeModel
from .LMTextRegressionModel import LMTextRegressionModel
from ..args import ModelArguments
from loguru import logger


class LMAutoModel:
    @classmethod
    def get_model(
        cls,
        model_args: ModelArguments,
        tune_strategy="none",
        ds_config=None,
        use_gpu: bool = True,
        *args,
        **kwargs,
    ):
        if model_args.arch_type == "decoder_only":
            return LMDecodeModel(
                model_args, tune_strategy, ds_config, use_gpu, *args, **kwargs
            )
        elif model_args.arch_type == "encoder_decoder":
            logger.warning("encoder_decoder has not support")
            return LMEncodeDecodeModel(
                model_args, tune_strategy, ds_config, use_gpu, *args, **kwargs
            )
        elif model_args.arch_type == "text_regression":
            logger.warning("encoder_decoder has not support")
            return LMTextRegressionModel(
                model_args, tune_strategy, ds_config, use_gpu, *args, **kwargs
            )
        else:
            raise NotImplementedError(f"暂不支持{model_args.arch_type}结构")
