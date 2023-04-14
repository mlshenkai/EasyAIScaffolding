# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/13 2:43 PM
# @File: Inferencer
# @Email: mlshenkai@163.com
import deepspeed
import torch.distributed as dist
from .base_pipeline import BasePipeline
from ..args import ModelArguments, DataArguments, InferenceArguments
from ..utils.data_utils import set_random_seed, batchlize
import os
import torch
from transformers import AutoConfig
from ..data.dataset import LMDataset
from ..models.base_model import LMBaseModel


class Inferencer(BasePipeline):
    """
    Inference
    """

    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        inference_args: InferenceArguments,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.inference_args = inference_args

        set_random_seed(self.inference_args.random_seed)

        self.local_rank = int(os.getenv("LOCAL_RABK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        if inference_args.device == "gpu":
            torch.cuda.set_device(self.local_rank)
            deepspeed.init_distributed()

        else:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "15000"
            dist.init_process_group(
                "gloo", rank=self.local_rank, world_size=self.world_size
            )
        self.config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code)
        try:
            self.model_hidden_size = self.config.hidden_size
        except:
            self.model_hidden_size = 1024

    def create_dataloader(self, dataset: LMDataset):
        data_dict = dataset.to_dict()
        inputs = [instance["text"] for instance in data_dict["instances"]]
        dataset_size = len(inputs)
        dataset_buf = []
        for idx in range(dataset_size):
            dataset_buf.append({"input": inputs[idx], "input_idx": idx})
        dataloader = batchlize(dataset_buf, batch_size=1, random_shuffle=False)
        return dataloader, dataset_size

    def inference(
        self,
        model: LMBaseModel,
        dataset: LMDataset,
        max_new_tokens: int,
        temperature: float = 0.8,
        prompt_structure: str = "{input}",
    ):
        """
        Perform inference for a model
        Args:
            model:
            dataset:
            max_new_tokens:
            temperature:
            prompt_structure:

        Returns:

        """
        if dataset.get_type() != "text_only":
            raise NotImplementedError("input dataset should have type text_only")
        dataloader, dataset_size = self.create_dataloader(dataset)

        output_dict = {
            "type": "text_only",
            "instances": [

            ]
        }

        for batch_idx, batch in enumerate(dataloader):
            current_batch = batch[0]  # we set batch size=1
            input = prompt_structure.format(input=current_batch["input"])
            if self.inference_args.device == "gpu":
                inputs = model.encode(input, return_tensors="pt").to(device=self.local_rank)
            else:
                inputs = model.encode(input, return_tensors="pt").to(device="cpu")
            outputs = model.inference(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            text_out = model.decode(outputs[0], skip_special_tokens=True)


            prompt_length = len(model.decode(inputs[0], skip_special_tokens=True))
            text_out = text_out[prompt_length:]
            output_dict["instances"].append(
                {
                    "text": text_out
                }
            )
        output_dataset = LMDataset(DataArguments(dataset_path=None))
        output_dataset = output_dataset.from_dict(output_dict)
        return output_dataset