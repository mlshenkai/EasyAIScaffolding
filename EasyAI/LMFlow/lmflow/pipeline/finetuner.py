# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/12 8:44 PM
# @File: finetuner
# @Email: mlshenkai@163.com
from itertools import chain
from pathlib import Path

from transformers.trainer_pt_utils import log_metrics, save_metrics, save_state

from .base_tuner import BaseTuner
from loguru import logger
import os
from transformers.utils import send_example_telemetry
from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed, Trainer, default_data_collator
import transformers.utils.logging
from ..args import ModelArguments, DataArguments, FinetunerArguments
from ..data.dataset import LLMDataset
from ..models.auto_model import HFAutoModel


class Finetuner(BaseTuner):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        finetuner_args: FinetunerArguments,
        *args,
        **kwargs,
    ):
        super(Finetuner, self).__init__()
        self.model_args = model_args
        self.data_args = data_args
        self.finetune_args = finetuner_args

        send_example_telemetry("run_clm", model_args, data_args)

        logger.info(f"Training/evaluation parameters {finetuner_args}")

        last_checkpoint = None
        if (
            os.path.isdir(finetuner_args.output_dir)
            and finetuner_args.do_train
            and not finetuner_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(finetuner_args.output_dir)
            if (
                last_checkpoint is None
                and len(os.listdir(finetuner_args.output_dir)) > 0
            ):
                raise ValueError(
                    f"Output dir {finetuner_args.output_dir} already exists and is not empty, please set overwrite_output_dir=True"
                )
            elif (
                last_checkpoint is not None
                and finetuner_args.resume_from_checkpoint is None
            ):
                logger.info(
                    f"Checkpoint detected, resuming training at"
                    f" {last_checkpoint}. To avoid this behavior, change"
                    " the `--output_dir` or add `--overwrite_output_dir` to"
                    " train from scratch."
                )
        self.last_checkpoint = last_checkpoint

        set_seed(finetuner_args.seed)

    def group_text(self, tokenized_dataset, model_max_length):
        """
        Groups texts together to form blocks of maximum length `model_max_length` and returns the processed data as
        a dictionary.
        Args:
            tokenized_dataset:
            model_max_length:

        Returns:

        """
        data_args = self.data_args
        finetune_args = self.finetune_args
        if data_args.block_size is None:
            block_size = model_max_length
            if block_size > 1024:
                logger.warning("max length must < 1024")
                block_size = 1024
        else:
            if data_args.block_size > model_max_length:
                logger.warning(
                    f"The block_size passed ({data_args.block_size}) is larger"
                    f" than the maximum length for the model"
                    f"({model_max_length})."
                    f" Using block_size={model_max_length}."
                )
            block_size = min(data_args.block_size, model_max_length)

        def group_texts(examples):
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model
            # supported it instead of this drop, you can customize this part to
            # your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        with finetune_args.main_process_first(desc="grouping texts together"):
            group_batch_size = 1000
            if data_args.disable_group_texts:
                group_batch_size = 1
            if not data_args.streaming:
                lm_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    batch_size=group_batch_size,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                lm_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    batch_size=group_batch_size,
                )

        return lm_datasets

    def tune(self, model: HFAutoModel, lm_dataset: LLMDataset):
        """
        Perform tuning for a model
        Args:
            model:
            lm_dataset:

        Returns:

        """
        model_args = self.model_args
        data_args = self.data_args
        finetune_args = self.finetune_args

        train_dataset = lm_dataset.get_backend_dataset()
        if finetune_args.do_train:
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

        train_args = finetune_args
        trainer = Trainer(
            model=model.get_backend_model(),
            args=train_args,
            train_dataset=train_dataset if train_args.do_train else None,
            eval_dataset=None,
            tokenizer=model.get_tokenizer(),
            data_collator=default_data_collator,
            compute_metrics=None,
            preprocess_logits_for_metrics=None,
        )
        if train_args.do_train:
            checkpoint = None
            last_checkpoint = self.last_checkpoint
            if train_args.resume_from_checkpoint is not None:
                checkpoint = train_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            if not model_args.use_lora:
                trainer.save_model()
            else:
                if model_args.save_aggregated_lora:
                    model.merge_lora_weight()
                model.save(
                    Path(finetune_args.output_dir), model_args.save_aggregated_lora
                )

            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            log_metrics(trainer, "train", metrics)
            save_metrics(trainer, "train", metrics)
            save_state(trainer)

        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "tasks": "text_generator",
        }

        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs[
                    "dataset"
                ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.create_model_card(**kwargs)
        return model
