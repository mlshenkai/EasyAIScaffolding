# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/14 3:03 PM
# @File: base_model
# @Email: mlshenkai@163.com
from abc import ABC

from transformers.testing_utils import CaptureLogger
import transformers.utils.logging
from ..args import ModelArguments
from ..data.dataset import LMDataset
from transformers.models.auto.auto_factory import _BaseAutoModelClass, PretrainedConfig
from transformers import AutoModel, AutoTokenizer, AutoConfig, CONFIG_MAPPING
from typing import Union, List, Any
from loguru import logger
from peft import PeftModel, LoraConfig, get_peft_config, get_peft_model
import deepspeed
import torch
import transformers
from peft import PeftConfig, LoraConfig, PeftModel, TaskType


class LMBaseModel(ABC):
    def __init__(
        self,
        model_args: ModelArguments,
        tune_strategy="none",
        ds_config=None,
        use_gpu: bool = True,
        *args,
        **kwargs,
    ):
        self.model_args = model_args
        self.tune_strategy = tune_strategy
        self.ds_config = ds_config
        self.use_gpu = use_gpu
        self.args = args
        self.kwargs = kwargs
        if self.tune_strategy == "none":
            # inference or valid process

            # 1. 判断是否有lora等prompt tuner 的模型文件
            peft_model_id = self.model_args.lora_model_path
            # llama 不可以使用ram方式加载
            if (
                "llama" in self.model_args.model_name_or_path
                and self.model_args.use_ram_optimized_load
            ):
                logger.warning(
                    "llama model does not support ram load. use original load instead"
                )
                self.model_args.use_ram_optimized_load = False

            if self.model_args.use_ram_optimized_load and peft_model_id is None:
                # 2. 尝试加载model, config, tokenizer
                try:
                    self.backend_model = self._load_backend_model(
                        model_name_or_path=self.model_args.model_name_or_path,
                        device_map="auto",
                        offload_folder="offload",
                        offload_state_dict=True,
                        trust_remote_code=self.model_args.trust_remote_code,
                    )
                except:
                    logger.warning(
                        "failed to use ram load model, use original load instead"
                    )
                    self.backend_model = self._load_backend_model(
                        model_name_or_path=self.model_args.model_name_or_path,
                        trust_remote_code=self.model_args.trust_remote_code,
                    )
            else:
                if peft_model_id is not None:
                    logger.warning("lora not support ram load")

                self.backend_model = self._load_backend_model(
                    model_name_or_path=self.model_args.model_name_or_path,
                    trust_remote_code=self.model_args.trust_remote_code,
                )
            self.tokenizer = self._load_backend_tokenizer(
                tokenizer_name_or_path=self.model_args.model_name_or_path,
                trust_remote_code=self.model_args.trust_remote_code,
            )
            self.backend_model_full = self.backend_model

            if peft_model_id is not None:
                self.backend_model = PeftModel.from_pretrained(
                    self.backend_model, peft_model_id
                )

            if self.use_gpu:
                deepspeed.init_distributed()
                self.ds_engine = deepspeed.initialize(
                    model=self.backend_model, config=self.ds_config
                )[0]
                self.ds_engine.module.eval()
        elif tune_strategy == "normal":
            # TODO split tune_strategy split [LoRA, P_tuning, Prompt_tuning, Prefix_tuning]
            model_config = {
                "revision": self.model_args.model_revision,
                "use_auth_token": True if self.model_args.use_auth_token else None,
                "use_fast": self.model_args.use_fast_tokenizer,
                "trust_remote_code": self.model_args.trust_remote_code,
            }
            if self.model_args.config_name:
                config = self._load_backend_config(
                    self.model_args.config_name, **model_config
                )
            elif self.model_args.model_name_or_path:
                config = self._load_backend_config(
                    self.model_args.model_name_or_path, **model_config
                )
            else:
                config: PretrainedConfig = CONFIG_MAPPING[self.model_args.model_type]()
                logger.warning(
                    "you not provide config_name or model_name, use original config"
                )
                if self.model_args.config_overrides is not None:
                    logger.info(f"Overate config: {self.model_args.config_overrides}")
                    config.update_from_string(self.model_args.config_overrides)
                    logger.info(f"New config: {config}")

            if self.model_args.tokenizer_name:
                tokenizer = self._load_backend_tokenizer(
                    self.model_args.tokenizer_name, **model_config
                )
            elif self.model_args.model_name_or_path:
                tokenizer = self._load_backend_tokenizer(
                    self.model_args.model_name_or_path, **model_config
                )
            else:
                raise ValueError("you not provide tokenizer_name or model_name")

            if self.model_args.model_name_or_path:
                torch_dtype = (
                    self.model_args.torch_dtype
                    if self.model_args.torch_dtype in ["auto", None]
                    else getattr(torch, model_args.torch_dtype)
                )
                model = self._load_backend_model(
                    model_name_or_path=self.model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                    config=config,
                    torch_dtype=torch_dtype,
                    **model_config,
                )
            else:
                model = AutoModel.from_config(config)

            self.backend_model_full = model

            if self.model_args.use_lora:
                peft_config = LoraConfig(
                    task_type=getattr(TaskType, self.model_args.lora_task_type),
                    inference_mode=False,
                    r=self.model_args.lora_r,
                    lora_alpha=self.model_args.lora_alpha,
                    lora_dropout=self.model_args.lora_dropout,
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            embedding_size = model.get_input_embeddings().weight.shape[0]
            if len(tokenizer) > embedding_size:
                model.resize_token_embeddings(len(tokenizer))

            self.model_args = model_args
            self.config = config
            self.backend_model = model
            self.tokenizer = tokenizer
            self.tune_strategy = tune_strategy
        else:
            raise NotImplementedError("please wait")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.padding_side = "left"

    def _load_backend_model(
        self, model_name_or_path, *args, **kwargs
    ) -> Union[Any, _BaseAutoModelClass]:
        return AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, **kwargs
        )

    def _load_backend_tokenizer(self, tokenizer_name_or_path, *args, **kwargs):
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_name_or_path, **kwargs
        )

    def _load_backend_config(
        self, config_name_or_path, *args, **kwargs
    ) -> Union[PretrainedConfig, Any]:
        return AutoConfig.from_pretrained(config_name_or_path, **kwargs)

    def tokenize(self, dataset: LMDataset, *args, **kwargs):
        """
        tokenizer the full dataset
        Args:
            dataset:
            *args:
            **kwargs:

        Returns:
            tokenized dataset
        """
        model_args = self.model_args
        row_datasets = dataset
        hf_row_datasets = dataset.get_backend_dataset()
        column_names = list(hf_row_datasets.features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        tok_logger = transformers.utils.logging.get_logger(
            "transformers.tokenization_utils_base"
        )

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                if not model_args.use_lora:
                    output = self.tokenizer(examples[text_column_name])
                else:
                    output = self.tokenizer(examples[text_column_name], truncation=True)

            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return output

        data_args = row_datasets.get_data_args()
        if not data_args.streaming:
            tokenized_datasets = row_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_fronm_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = row_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )
        return tokenized_datasets

    def encode(self, inputs: Union[str, List[str]], *args, **kwargs):
        """
        Perform encoding process of the tokenizer.
        Args:
            inputs:
            *args:
            **kwargs:

        Returns:
            if input is string, return tokenized input string,
            if input is list, return input_ids, attention_mask, token_type_ids
        """
        if isinstance(input, str):
            return self.tokenizer.encode(text=input, *args, **kwargs)
        elif isinstance(input, list):
            return self.tokenizer(text=input, *args, **kwargs)

    def decode(self, outputs, *args, **kwargs) -> Union[str, List[str]]:
        """
        Perform decoding process of the tokenizer.
        Args:
            input:
            *args:
            **kwargs:

        Returns:
            the text decoded from the token
        """

        if isinstance(input, list) and input and isinstance(input[0], list):
            output = []
            for single_input in input:
                single_output = self.decode(single_input, *args, **kwargs)
                output.append(single_output)
            return output
        else:
            # Can be list of ints or a Tensor
            return self.tokenizer.decode(input, *args, **kwargs)

    def generate(self, inputs, *args, **kwargs):
        raise NotImplementedError("inference func must be implement")

    @torch.no_grad()
    def inference(self, inputs, *args, **kwargs):
        if self.use_gpu:
            outputs = self.backend_model.module(inputs, **kwargs)
            return outputs
        else:
            outputs = self.backend_model(inputs, **kwargs)
            return outputs


    def merge_tune_weight(self, *args, **kwargs):
        if self.model_args.use_lora:
            self.get_backend_model().merge_and_upload()
        else:
            logger.warning(
                "LoRA training is NOT enabled. Merging LoRA weights is not applicable."
            )

    def save(self, dir: str, save_full_name=False, *args, **kwargs):
        self.get_tokenizer().save_pretrained(dir)
        if save_full_name and self.model_args.use_lora:
            self.backend_model_full.save_pretrained(dir)
        else:
            self.get_backend_model().save_pretrained(dir)

    def get_max_length(self):
        return self.tokenizer.model_max_length

    def get_tokenizer(self):
        return self.tokenizer

    def get_backend_model(self):
        return self.backend_model

    def get_backend_model_full(self):
        return self.backend_model_full
