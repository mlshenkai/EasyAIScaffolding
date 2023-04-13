# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/12 3:52 PM
# @File: auto_model
# @Email: mlshenkai@163.com
from loguru import logger
from typing import List, Union
from ..data.dataset import LLMDataset
from pathlib import Path
import deepspeed
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_config,
    prepare_model_for_int8_training,
)

# from peft import PromptTuningConfig, PEFT_TYPE_TO_CONFIG_MAPPING
import torch
import transformers
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.testing_utils import CaptureLogger
import transformers.utils.logging
from transformers import CONFIG_MAPPING, AutoConfig, AutoModel, AutoTokenizer
from dataclasses import make_dataclass, asdict

from ..args import ModelArguments


class HFAutoModel:
    """
    init model
    """

    def __init__(
        self,
        model_args: ModelArguments,
        tune_strategy="none",
        ds_config=None,
        use_gpu: bool = True,
        *args,
        **kwargs,
    ):
        self.use_gpu = use_gpu
        # merged_config_dict = {**asdict(model_args), **kwargs}
        # UpdatedModelConfig = make_dataclass("UpdatedModelConfig", merged_config_dict.keys())
        # model_args = UpdatedModelConfig(**merged_config_dict)
        model_kwargs = {**asdict(model_args), **kwargs}
        if tune_strategy == "none":
            # infer or eval stage
            peft_model_id = model_args.lora_model_path
            if (
                "llama" in model_args.model_name_or_path
                and model_args.use_ram_optimized_load
            ):
                logger.warning(
                    "llama dest not support ram load, use original load instead"
                )
                model_args.use_ram_optimized_load = False
            if model_args.use_ram_optimized_load and peft_model_id is None:
                try:
                    # 尝试使用ram 加载 模型
                    self.backend_model = AutoModel.from_pretrained(
                        model_args.model_name_or_path,
                        device_map="auto",
                        offload_folder="offload",
                        offload_state_dict=True,
                        **model_kwargs,
                    )
                except:
                    logger.warning(
                        "Failed to use RAM load model \n" "use original load instead "
                    )

                    self.backend_model = AutoModel.from_pretrained(
                        model_args.model_name_or_path, **model_kwargs
                    )
            else:
                if peft_model_id is not None:
                    logger.warning(
                        "LoRA not support RAM load, use original load instead"
                    )
                self.backend_model = AutoModel.from_pretrained(
                    model_args.model_name_or_path, **model_kwargs
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path
            )
            self.backend_model_full = self.backend_model
            if peft_model_id is not None:
                self.backend_model = PeftModel.from_pretrained(
                    self.backend_model, peft_model_id
                )
            if self.use_gpu:
                deepspeed.init_distributed()
                self.ds_engine = deepspeed.initialize(
                    model=self.backend_model, config_params=ds_config
                )[0]
                self.ds_engine.modules().eval()
        elif tune_strategy == "normal":
            # TODO split tune_strategy split [LoRA, P_tuning, Prompt_tuning, Prefix_tuning]
            # now, just support LoRA, it most simple for us
            model_kwargs["revision"] = model_kwargs["model_revision"]
            model_kwargs["use_auth_token"] = (
                True if model_kwargs.get("use_auth_token") else None
            )
            model_kwargs["use_fast"] = model_args.use_fast_tokenizer
            if model_args.config_name:
                config = AutoConfig.from_pretrained(
                    model_args.config_name, **model_kwargs
                )
            elif model_args.model_name_or_path:
                config = AutoConfig.from_pretrained(
                    model_args.model_name_or_path, **model_kwargs
                )
            else:
                config = CONFIG_MAPPING[model_args.model_type]()
                logger.warning(
                    "not provide config_name and model_name_or_path, use original config instead"
                )
                if model_args.config_overrides is not None:
                    logger.info(f"Overriding config: {model_args.config_overrides}")
                    config.update_from_string(model_args.config_overrides)
                    logger.info(f"New config: {config}")
            if model_args.tokenizer_name:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_args.tokenizer_name, **model_kwargs
                )
            elif model_args.model_name_or_path:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_args.model_name_or_path, **model_kwargs
                )
            else:
                raise ValueError(
                    "You are instantiating a new tokenizer from scratch. This is"
                    " not supported by this script. You can do it from another"
                    " script, save it, and load it from here, using"
                    " --tokenizer_name."
                )
            if model_args.model_name_or_path:
                torch_dtype = (
                    model_args.torch_dtype
                    if model_args.torch_dtype in ["auto", None]
                    else getattr(torch, model_args.torch_dtype)
                )
                model = AutoModel.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    torch_dtype=torch_dtype,
                    **model_kwargs,
                )
            else:
                model = AutoModel.from_config(config)
            self.backend_model_full = model

            if model_args.use_lora:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
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
        elif tune_strategy == "adapter":
            raise NotImplementedError("adapter be not implement")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.aos_token_id
        self.tokenizer.padding_side = "left"

    def tokenize(self, dataset: LLMDataset, *args, **kwargs):
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

    def encode(self, input: Union[str, List[str]], *args, **kwargs):
        """
        Perform encoding process of the tokenizer.
        Args:
            input:
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

    def decode(self, input, *args, **kwargs) -> Union[str, List[str]]:
        """
        Perform decoding process of the tokenizer.
        Args:
            input:
            *args:
            **kwargs:

        Returns:
            the text decoded from the token
        """

        if isinstance(input, List):
            input = torch.tensor(input)
        if input.dim() == 2:
            return self.tokenizer.batch_decode(input, *args, **kwargs)
        else:
            return self.tokenizer.decode(input, *args, **kwargs)

    def inference(self, inputs, *args, **kwargs):
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
                    pad_token_id=self.tokenizer.pad_token_id,
                    *args,
                    **kwargs,
                )
            else:
                outputs = self.backend_model.generator(
                    input_ids=inputs,
                    synced_gpus=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    *args,
                    **kwargs,
                )
        return outputs

    def merge_lora_weight(self):
        if self.model_args.use_lora:
            self.get_backend_model().merge_and_upload()
        else:
            logger.warning(
                "LoRA training is NOT enabled. Merging LoRA weights is not applicable."
            )

    def save(self, dir: Path, save_full_name=False, *args, **kwargs):
        self.get_tokenizer().save_pretrained(dir.as_posix())
        if save_full_name and self.model_args.use_lora:
            self.backend_model_full.save_pretrained(dir.as_posix())
        else:
            self.get_backend_model().save_pretrained(dir.as_posix())

    def get_max_length(self):
        return self.tokenizer.model_max_length

    def get_tokenizer(self):
        return self.tokenizer

    def get_backend_model(self):
        return self.backend_model
