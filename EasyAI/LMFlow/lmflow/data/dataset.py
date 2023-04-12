# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/12 11:57 AM
# @File: dataset
# @Email: mlshenkai@163.com

from datasets import load_dataset, Dataset
from typing import Optional
from ..args import DataArguments
from pathlib import Path
from .base_dataset import LLMBaseDataset
import json
import csv


DATASET_TYPES = {"text_only", "text2text"}
KEY_TYPE = "type"
KEY_INSTANCES = "instances"


class LLMDataset(LLMBaseDataset):
    def __init__(
        self,
        data_args: DataArguments = None,
        backend: str = "huggingface",
        *args,
        **kwargs,
    ):
        super(LLMDataset, self).__init__(data_args, backend, *args, **kwargs)
        if backend == "huggingface":
            data_files = [
                x.absolute().as_posix() for x in Path(self.dataset_path).glob("*.json")
            ]

            # Iterate through all the files and ensure they have the same data type
            for single_file in data_files:
                with open(single_file) as fin:
                    json_data = json.load(fin)
                    if KEY_TYPE not in json_data.keys():
                        raise ValueError(
                            f'"{KEY_TYPE}" field must be specified for data, e.g.'
                            "{\n"
                            f'   "{KEY_TYPE}: "text_only",\n'
                            f'   "{KEY_INSTANCES}": [\n'
                            '       { "text": "Sentence 1: This is a sentence." }\n'
                            '       { "text": "Sentence 2: This is another sentence." }\n'
                            f"   ]\n"
                            "}"
                        )

                    if self.type is None:
                        self.type = json_data[KEY_TYPE]
                    elif self.type != json_data[KEY_TYPE]:
                        raise ValueError(
                            "All task files must have same data types. Previous"
                            f' files have type "{self.type}", but in file'
                            f' {single_file}, it has type "{self.type}".'
                        )

            # Load the dataset using the HuggingFace dataset library
            extensions = "json"
            raw_dataset = load_dataset(
                extensions,
                data_files=data_files,
                field=KEY_INSTANCES,
                split="train",
                use_auth_token=None,
            )
            self.backend_dataset = raw_dataset
        elif backend == "json":
            # TODO (@watcher)
            pass
        elif backend == "csv":
            # TODO (@watcher)
            pass
        else:
            raise NotImplementedError(f'Unsupported dataset backend "{backend}"')
