# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/12 2:14 PM
# @File: base_dataset
# @Email: mlshenkai@163.com
from abc import ABC
from collections import defaultdict
from datasets import Dataset as HFDataset
from ..args import DataArguments


class LLMBaseDataset(ABC):
    def __init__(self, data_args: DataArguments = None, backend: str = "huggingface", *args, **kwargs):
        self.data_args = data_args
        self.backend = backend
        self.backend_dataset = None
        self.type = None
        if self.data_args is None:
            return
        self.dataset_path = data_args.dataset_path
        if self.dataset_path is None:
            return

    def _check_data_structure(self, data_obj: dict):
        """
        check data structure is match
        Returns:

        """
        if self.backend != "huggingface":
            raise NotImplementedError(
                f" if {self.backend} != huggingface, you must implement this func to build yourself dataset."
            )
        else:
            if "type" not in data_obj:
                raise ValueError("type must be provided")
            if "instances" not in data_obj:
                raise ValueError("instances must be provided")

    def from_dict(self, data_obj: dict, *args, **kwargs):
        """

        Args:
            data_obj:
                data_obj must belong to huggingface
                {
                    "type": TYPE,
                    "instances":[
                        {
                            "key_1": VALUE_1.1,
                            "key_2": VALUE_1.2,
                            ...
                        },
                        {
                            "key_1": VALUE_2.1,
                            "key_2": VALUE_2.2,
                            ...

                        },
                        ...

                    ]
                }
            *args:
            **kwargs:

        Returns:
            Dataset instances
        """
        if self.backend != "huggingface":
            raise NotImplementedError(
                f" if {self.backend} != huggingface, you must implement this func to build yourself dataset."
            )
        else:
            self._check_data_structure(data_obj)
            self.type = data_obj["type"]
            hf_dict = defaultdict(list)
            if len(data_obj["instances"]) > 0:
                instances = data_obj["instances"]
                for instance in instances:
                    for key in list(instance.keys()):
                        value = instance[key]
                        hf_dict[key].append(value)
            self.backend_dataset = HFDataset.from_dict(hf_dict, *args, **kwargs)

    @classmethod
    def create_from_dict(cls, dict_obj, *args, **kwargs):
        r"""
        Returns
        --------

        Returns a Dataset object given a dict.
        """
        empty_data_args = DataArguments(dataset_path=None)
        dataset = cls(empty_data_args)
        return dataset.from_dict(dict_obj)

    def to_dict(self):
        """
        from backend_dataset to dict
        Returns:
            {
                    "type": TYPE,
                    "instances":[
                        {
                            "key_1": VALUE_1.1,
                            "key_2": VALUE_1.2,
                            ...
                        },
                        {
                            "key_1": VALUE_2.1,
                            "key_2": VALUE_2.2,
                            ...

                        },
                        ...

                    ]
                }
        """
        if self.backend != "huggingface":
            raise NotImplementedError(
                f" if {self.backend} != huggingface, you must implement this func"
            )
        else:
            data_obj = {}
            data_obj["type"] = self.get_type()
            data_obj["instances"] = []
            hf_dict = self.backend_dataset.to_dict()

            first_key = None
            for key in hf_dict.keys():
                first_key = key
                break

            if first_key is not None:
                num_instances = len(hf_dict[first_key])
                data_obj["instances"] = [
                    {key: hf_dict[key][i] for key in hf_dict.keys()}
                    for i in range(num_instances)
                ]

            return data_obj

    def get_type(self):
        if self.backend != "huggingface":
            raise NotImplementedError(
                f" if {self.backend} != huggingface, you must implement this func"
            )
        else:
            return self.type

    def get_backend(self):
        if self.backend != "huggingface":
            raise NotImplementedError(
                f" if {self.backend} != huggingface, you must implement this func"
            )
        else:
            return self.backend

    def map(self, *args, **kwargs):
        if self.backend != "huggingface":
            mapped_backend_dataset = self.backend_dataset.map(*args, **kwargs)
            self.backend_dataset = mapped_backend_dataset
            return self

        else:
            raise NotImplementedError(
                f" if {self.backend} != huggingface, you must implement this func"
            )

    def get_backend_dataset(self) -> HFDataset:
        return self.backend_dataset

    def get_data_args(self):

        return self.data_args
