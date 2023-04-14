# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/4/13 1:48 PM
# @File: raft_aligner
# @Email: mlshenkai@163.com
from itertools import chain
import numpy as np
import torch
import time
import torch.distributed as dist
from transformers.testing_utils import CaptureLogger
from ..data.dataset import LMDataset
from .base_aligner import BaseAligner
from ..args import ModelArguments, DataArguments, RaftAlignerArguments, TrainingArguments
import os
import transformers
import transformers.utils.logging
from loguru import logger
from .utils.raft_trainer import RaftTrainer
from datasets import Dataset, DatasetDict, set_caching_enabled
from transformers import default_data_collator, set_seed


class RaftAligner(BaseAligner):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        raft_aligner_args: RaftAlignerArguments,
        *args,
        **kwargs
    ):
        super(RaftAligner, self).__init__(*args, **kwargs)
        self.model_args = model_args
        self.data_args = data_args
        self.aligner_args = raft_aligner_args
        output_reward_path = raft_aligner_args.output_reward_path
        if output_reward_path is not None:
            os.makedirs(os.path.dirname(output_reward_path), exist_ok=True)
            try:
                os.remove(output_reward_path)
            except OSError:
                pass

    def _init_trainer(self, model, tokenizer, training_args):
        trainer = RaftTrainer(
            model=model,
            args=training_args,
            train_dataset=Dataset.from_dict({}),
            eval_dataset=Dataset.from_dict({}),
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=None,
            preprocess_logits_for_metrics=None
        )
        return trainer

    def _load_dataset(self, selected_dataset, model, tokenizer, model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
        raw_datasets = selected_dataset
        if training_args.do_train:
            column_names = list(raw_datasets["train"].features)
        else:
            column_names = list(raw_datasets["validation"].features)

        text_column_name = "text" if "text" in column_names else column_names[0]
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer(examples[text_column_name])
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return output

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

        if data_args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 1024:
                logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )
                block_size = 512
        else:
            if data_args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(data_args.block_size, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with training_args.main_process_first(desc="grouping texts together"):
            group_batch_size = 1000
            if data_args.disable_group_texts:
                group_batch_size = 1
            if not data_args.streaming:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    batch_size=group_batch_size,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    batch_size=group_batch_size,
                )

        if training_args.do_train:
            if "train" not in tokenized_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = lm_datasets["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

        return train_dataset

    def _load_input_dataset(self, dataset, tokenizer):
        """
        laod input dataset (i.e. prompt/question) for training
        Args:
            dataset:
            tokenizer:

        Returns:

        """
        ds = dataset.get_backend_dataset()

        def tokenize(sample):
            input_size = 16
            review_encode = tokenizer.encode(sample["text"])
            sample["input_ids"] = review_encode[:input_size]
            sample['input'] = tokenizer.decode(sample["input_ids"])
            return sample
        ds = ds.map(tokenize, batched=False)
        ds.set_format(type='torch')
        return ds

    def _get_batch_dataset_top(
            self,
            model,
            batch_input,
            alpha=0.2,
            iter_id=0,
            local_rank=0,
            output_min_length=16,
            output_max_length=48,
            infer_batch_size=8,
            generation_kwargs={},
            tokenizer=None,
            training_args=None,
            reward_model=None,
            output_reward_path=None,
    ):
        """
        :param batch_input: input prompts
        """
        # we will get the batch dataset via Dataset.from_dict
        start_time = time.time()
        output_data = []
        query_tensors = batch_input['input_ids']
        querys = batch_input['input']
        data_size = len(querys)
        cnt = 0
        reward_eva = []
        reward_train = []
        out_put_dataset_eval = {}
        data_eval = []
        input_texts = []
        responses = []
        for i, query_tensor in enumerate(query_tensors):
            query = querys[i]
            input_texts.append(query)
            if (i + 1) % infer_batch_size == 0:
                gen_len = np.random.randint(output_min_length, output_max_length)
                generation_kwargs["max_new_tokens"] = gen_len
                inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(training_args.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, **generation_kwargs)
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                generated_texts = [
                    generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
                ]
                texts_for_rewards = [q + r for q, r in zip(input_texts, generated_texts)]

                texts_for_reward_dataset = LMDataset.create_from_dict({
                    "type": "text_only",
                    "instances": [
                        { "text": text } for text in texts_for_rewards
                    ],
                })

                reward_dataset = reward_model.get_regression(texts_for_reward_dataset)
                rewards = [ sample["value"] for sample in reward_dataset.to_dict()["instances"] ]

                reward_eva.extend(rewards)
                responses.extend(generated_texts)
                input_texts = []

        data = []
        idx = np.argsort(reward_eva)[::-1][:int(data_size * alpha)]
        for j in range(len(reward_eva)):
            sample = {}
            sample["input"] = querys[j]
            sample["output"] = [responses[j]]
            data.append(sample)
        output_data = [data[j] for j in idx]
        logger.info(f"collected data of {len(output_data)}")

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        all_process_list =[{}] * world_size
        dist.all_gather_object(all_process_list, output_data)

        gathered_data = []
        for i in range(world_size):
            gathered_data.extend(all_process_list[i])

        reward_train = [reward_eva[j] for j in idx]

        reward_to_send = [np.mean(reward_eva), np.mean(reward_train)]
        all_process_rewards = [{}] * world_size
        dist.all_gather_object(all_process_rewards, reward_to_send)
        logger.info(all_process_rewards)

        if training_args.local_rank == 0 and output_reward_path is not None:
            with open(output_reward_path, mode='a') as fout:
                fout.write('mean reward: ' + str(np.mean([all_process_rewards[i][0] for i in range(world_size)])) + 'mean reward in training set: ' + str([all_process_rewards[i][1] for i in range(world_size)]))
                fout.write("\n")

        prompt_structure = "{definition}{input}{output}"
        output_dataset = {
            "text": [ prompt_structure.format(
                definition="", input=sample["input"], output=sample["output"][0]
            ) for sample in gathered_data
            ]
        }

        return DatasetDict({ "train": Dataset.from_dict(output_dataset) })

    def align(self, model, dataset, reward_model):
        """
        Perform alignment for model
        Args:
            model:
            dataset:
            reward_model:

        Returns:

        """
        tokenizer = model.get_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        dataset = self._load_input_dataset(dataset, tokenizer)

        set_caching_enabled(False)

        wrapped_model = model
        model = model.get_backend_model()

        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "temperature":0.7
        }
        aligner_args = self.aligner_args
        training_args = aligner_args
        model_args = self.model_args
        data_args = self.data_args

        set_seed(42 + training_args.local_rank)
        ITERATION = aligner_args.num_raft_iteration
        M = aligner_args.raft_batch_size

        alpha = aligner_args.top_reward_percentage
        data_size = len(dataset['input'])
        reward_seq = []
        lr = training_args.learning_rate

        raft_trainer = self._initialize_trainer(model, tokenizer, training_args)
        raft_trainer.train(resume_from_checkpoint=False, is_first_time=True)

        ##############
        for iteration in range(ITERATION):
            set_seed(88 + training_args.local_rank + 4 * (iteration+1))

            batch_input = dataset.select(np.random.randint(low=0, high=data_size, size=M))

            selected_dataset = self._get_batch_dataset_top(
                raft_trainer.tmp_model,
                batch_input,
                alpha,
                iteration,
                training_args.local_rank,
                output_min_length=aligner_args.output_min_length,
                output_max_length=aligner_args.output_max_length,
                infer_batch_size=aligner_args.inference_batch_size_per_device,
                generation_kwargs=generation_kwargs,
                tokenizer=tokenizer,
                training_args=training_args,
                reward_model=reward_model,
                output_reward_path=aligner_args.output_reward_path,
            )
            raft_trainer.train_dataset = self._load_dataset(
                selected_dataset,
                raft_trainer.tmp_model,
                tokenizer,
                model_args,
                data_args,
                training_args,
            )

            logger.info(f"iter {iteration}")
            start_time = time.time()
            train_result = raft_trainer.train(resume_from_checkpoint=False)
            end_time = time.time()
            logger.info("It takes %.2f s to train one stage", end_time - start_time)

            self._get_batch_dataset_top(
                raft_trainer.tmp_model,
                batch_input,
                alpha,
                iteration,
                training_args.local_rank,
                output_min_length=aligner_args.output_min_length,
                output_max_length=aligner_args.output_max_length,
                infer_batch_size=aligner_args.inference_batch_size_per_device,
                generation_kwargs=generation_kwargs,
                tokenizer=tokenizer,
                training_args=training_args,
                reward_model=reward_model,
                output_reward_path=aligner_args.output_reward_path,
            )

        if aligner_args.output_dir is not None:
            wrapped_model.save(aligner_args.output_dir)

        return wrapped_model

