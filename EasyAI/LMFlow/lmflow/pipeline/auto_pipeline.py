# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/13 3:17 PM
# @File: auto_pipeline
# @Email: mlshenkai@163.com
from .finetuner import Finetuner
from .inferencer import Inferencer
from .raft_aligner import RaftAligner


PIPELINE_MAPPING = {
    "finetuner": Finetuner,
    "inference": Inferencer,
    "raft_aligner": RaftAligner,
}


class AutoPipeline:
    """
    The class designed to return a pipeline automatically based on its name.
    """

    @classmethod
    def get_pipeline(
        self, pipeline_name, model_args, data_args, pipeline_args, *args, **kwargs
    ):
        if pipeline_name not in PIPELINE_MAPPING:
            raise NotImplementedError(f'Pipeline "{pipeline_name}" is not supported')

        pipeline = PIPELINE_MAPPING[pipeline_name](
            model_args, data_args, pipeline_args, *args, **kwargs
        )
        return pipeline
